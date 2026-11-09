import asyncio
import datetime
import threading
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from docling.datamodel.base_models import DocumentStream
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.tasks import TaskType
from docling_jobkit.convert.manager import DoclingConverterManager
from docling_jobkit.datamodel.task import Task
from docling_jobkit.datamodel.task_meta import TaskStatus

from docling_serve.orchestrator_factory import get_async_orchestrator
from docling_serve.settings import AsyncEngine, docling_serve_settings
from docling_serve.stuck_task_watchdog import (
    STUCK_TASK_ERROR_MESSAGE,
    STUCK_TASK_HEALTH_DETAIL,
    LocalStuckTaskWatchdog,
    task_deadline_seconds,
)

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _source(name: str = "a.pdf") -> DocumentStream:
    return DocumentStream(name=name, stream=BytesIO(b"%PDF-1.4"))


def _task(
    task_id: str = "t1",
    document_timeout: float | None = 10.0,
    num_sources: int = 1,
    status: TaskStatus = TaskStatus.STARTED,
    started_at: datetime.datetime | None = T0,
) -> Task:
    task = Task(
        task_id=task_id,
        task_type=TaskType.CONVERT,
        sources=[_source(f"{i}.pdf") for i in range(num_sources)],
        convert_options=ConvertDocumentsOptions(document_timeout=document_timeout),
    )
    task.task_status = status
    task.started_at = started_at
    task.last_update_at = started_at or T0
    return task


def _orchestrator(*tasks: Task) -> SimpleNamespace:
    notifier = SimpleNamespace(notify_task_subscribers=AsyncMock())
    return SimpleNamespace(
        tasks={task.task_id: task for task in tasks},
        notifier=notifier,
        _task_results={},
    )


def _at(seconds: float) -> datetime.datetime:
    return T0 + datetime.timedelta(seconds=seconds)


def test_deadline_is_document_timeout_per_source_plus_grace():
    assert task_deadline_seconds(_task(document_timeout=10), 5) == 15
    assert task_deadline_seconds(_task(document_timeout=10, num_sources=3), 5) == 35


def test_no_deadline_without_document_timeout():
    assert task_deadline_seconds(_task(document_timeout=None), 5) is None


@pytest.mark.asyncio
async def test_task_within_deadline_is_left_alone():
    task = _task(document_timeout=10, num_sources=2)
    orchestrator = _orchestrator(task)
    watchdog = LocalStuckTaskWatchdog(orchestrator, 5, 1)

    assert await watchdog.check(now=_at(25)) == []
    assert task.task_status == TaskStatus.STARTED
    assert not watchdog.has_stuck_tasks()
    orchestrator.notifier.notify_task_subscribers.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_past_deadline_is_failed_and_reported_stuck():
    task = _task(document_timeout=10)
    orchestrator = _orchestrator(task)
    watchdog = LocalStuckTaskWatchdog(orchestrator, 5, 1)

    failed = await watchdog.check(now=_at(15.5))

    assert failed == [task]
    assert task.task_status == TaskStatus.FAILURE
    assert task.error_message == STUCK_TASK_ERROR_MESSAGE
    assert task.finished_at is not None
    assert watchdog.stuck_task_ids == ["t1"]
    orchestrator.notifier.notify_task_subscribers.assert_awaited_once_with("t1")

    # A second scan does not fail or notify again.
    assert await watchdog.check(now=_at(100)) == []
    orchestrator.notifier.notify_task_subscribers.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_and_finished_tasks_are_ignored():
    pending = _task("p", status=TaskStatus.PENDING, started_at=None)
    done = _task("d", status=TaskStatus.SUCCESS)
    orchestrator = _orchestrator(pending, done)
    watchdog = LocalStuckTaskWatchdog(orchestrator, 0, 1)

    assert await watchdog.check(now=_at(10_000)) == []
    assert pending.task_status == TaskStatus.PENDING
    assert done.task_status == TaskStatus.SUCCESS


@pytest.mark.asyncio
async def test_late_success_from_abandoned_thread_is_discarded():
    task = _task(document_timeout=10)
    orchestrator = _orchestrator(task)
    watchdog = LocalStuckTaskWatchdog(orchestrator, 0, 1)
    await watchdog.check(now=_at(11))

    # The wedged thread eventually returns and the local worker records a result.
    orchestrator._task_results["t1"] = object()
    task.set_status(TaskStatus.SUCCESS)

    await watchdog.check()

    assert not watchdog.has_stuck_tasks()
    assert task.task_status == TaskStatus.FAILURE
    assert task.error_message == STUCK_TASK_ERROR_MESSAGE
    assert "t1" not in orchestrator._task_results
    assert orchestrator.notifier.notify_task_subscribers.await_count == 2


@pytest.mark.asyncio
async def test_late_failure_keeps_the_stuck_error_message():
    task = _task(document_timeout=10)
    orchestrator = _orchestrator(task)
    watchdog = LocalStuckTaskWatchdog(orchestrator, 0, 1)
    await watchdog.check(now=_at(11))

    # The worker's except branch overwrites the message when the thread raises.
    task.set_status(TaskStatus.FAILURE)
    task.error_message = "Internal error"

    await watchdog.check()

    assert not watchdog.has_stuck_tasks()
    assert task.task_status == TaskStatus.FAILURE
    assert task.error_message == STUCK_TASK_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_thread_still_running_keeps_task_stuck():
    task = _task(document_timeout=10)
    orchestrator = _orchestrator(task)
    watchdog = LocalStuckTaskWatchdog(orchestrator, 0, 1)
    await watchdog.check(now=_at(11))

    await watchdog.check(now=_at(500))

    assert watchdog.stuck_task_ids == ["t1"]


@pytest.mark.asyncio
async def test_released_after_client_deleted_the_task():
    task = _task(document_timeout=10)
    orchestrator = _orchestrator(task)
    watchdog = LocalStuckTaskWatchdog(orchestrator, 0, 1)
    await watchdog.check(now=_at(11))
    notified = orchestrator.notifier.notify_task_subscribers.await_count

    del orchestrator.tasks["t1"]
    task.set_status(TaskStatus.SUCCESS)
    await watchdog.check()

    assert not watchdog.has_stuck_tasks()
    assert orchestrator.notifier.notify_task_subscribers.await_count == notified


@pytest.mark.asyncio
async def test_notifier_errors_do_not_break_the_scan():
    task = _task(document_timeout=10)
    orchestrator = _orchestrator(task)
    orchestrator.notifier.notify_task_subscribers.side_effect = RuntimeError("ws")
    watchdog = LocalStuckTaskWatchdog(orchestrator, 0, 1)

    assert await watchdog.check(now=_at(11)) == [task]
    assert task.task_status == TaskStatus.FAILURE


# End to end: a conversion that never returns, served by the real app.


@pytest.fixture
def wedged_conversions(monkeypatch):
    """Make every conversion block its worker thread until released."""
    release = threading.Event()
    entered = threading.Event()

    def convert_documents(self, *args, **kwargs):
        entered.set()
        release.wait()
        raise RuntimeError("released")

    monkeypatch.setattr(DoclingConverterManager, "convert_documents", convert_documents)
    yield SimpleNamespace(release=release, entered=entered)
    release.set()


@pytest.fixture
def watchdog_settings(monkeypatch):
    # Each test runs on its own event loop, so it needs its own orchestrator
    # (the local engine's asyncio.Queue is bound to the loop that first used it).
    get_async_orchestrator.cache_clear()
    yield from _patch_settings(monkeypatch)
    get_async_orchestrator.cache_clear()


def _patch_settings(monkeypatch):
    monkeypatch.setattr(docling_serve_settings, "eng_kind", AsyncEngine.LOCAL)
    monkeypatch.setattr(docling_serve_settings, "load_models_at_boot", False)
    monkeypatch.setattr(docling_serve_settings, "eng_loc_stuck_task_grace_s", 0.5)
    monkeypatch.setattr(
        docling_serve_settings, "eng_loc_stuck_task_check_interval_s", 0.1
    )
    monkeypatch.setattr(docling_serve_settings, "eng_loc_unhealthy_on_stuck_task", True)
    yield


async def _wait_for(predicate, timeout: float = 10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await predicate()
        if result:
            return result
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_wedged_conversion_is_failed_and_liveness_recovers(
    watchdog_settings, wedged_conversions
):
    from docling_serve.app import create_app

    app = create_app()
    try:
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://app.io"
            ) as client:
                response = await client.post(
                    "/v1/convert/source/async",
                    json={
                        "options": {"document_timeout": 0.5},
                        "sources": [
                            {
                                "kind": "file",
                                "base64_string": "aGVsbG8=",
                                "filename": "wedged.pdf",
                            }
                        ],
                    },
                )
                assert response.status_code == 200, response.text
                task_id = response.json()["task_id"]

                async def status():
                    r = await client.get(f"/v1/status/poll/{task_id}")
                    return r.json()

                async def failed():
                    data = await status()
                    return data if data["task_status"] == "failure" else None

                # Without the watchdog this task would stay "started" forever.
                data = await _wait_for(failed)
                assert wedged_conversions.entered.is_set()
                assert data["error_message"] == STUCK_TASK_ERROR_MESSAGE

                livez = await client.get("/livez")
                assert livez.status_code == 503
                assert livez.json()["detail"] == STUCK_TASK_HEALTH_DETAIL
                assert (await client.get("/ready")).status_code == 503
                # /health stays a plain process check.
                assert (await client.get("/health")).status_code == 200

                # Once the wedged thread returns, the worker is free again.
                wedged_conversions.release.set()

                async def live():
                    return (await client.get("/livez")).status_code == 200

                await _wait_for(live)
                data = await status()
                assert data["task_status"] == "failure"
                assert data["error_message"] == STUCK_TASK_ERROR_MESSAGE
    finally:
        # Never leave a blocked thread behind, or the event loop cannot close.
        wedged_conversions.release.set()


@pytest.mark.asyncio
async def test_unhealthy_flag_off_only_fails_the_task(
    watchdog_settings, wedged_conversions, monkeypatch
):
    from docling_serve.app import create_app

    monkeypatch.setattr(
        docling_serve_settings, "eng_loc_unhealthy_on_stuck_task", False
    )
    app = create_app()
    async with LifespanManager(app) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://app.io"
        ) as client:
            response = await client.post(
                "/v1/convert/source/async",
                json={
                    "options": {"document_timeout": 0.5},
                    "sources": [
                        {
                            "kind": "file",
                            "base64_string": "aGVsbG8=",
                            "filename": "wedged.pdf",
                        }
                    ],
                },
            )
            task_id = response.json()["task_id"]

            async def failed():
                r = await client.get(f"/v1/status/poll/{task_id}")
                return r.json()["task_status"] == "failure"

            try:
                await _wait_for(failed)
                assert (await client.get("/livez")).status_code == 200
                assert (await client.get("/ready")).status_code == 200
            finally:
                wedged_conversions.release.set()
