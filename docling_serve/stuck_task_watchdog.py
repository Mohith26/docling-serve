"""Fail local-engine tasks whose conversion outlives its deadline.

The local engine runs every conversion in a worker thread
(``asyncio.to_thread``). Docling checks ``document_timeout`` between pages, so a
conversion that wedges inside a single call (a native parser or OCR call that
never returns, a hung download) is never interrupted: the task stays
``started`` forever, its worker slot is lost, and once every slot is lost the
server accepts tasks it will never run while ``/livez`` keeps reporting ok.

Python cannot stop a running thread, so this watchdog does the next best thing:
it fails the task (so clients waiting on it get an answer) and, while the
abandoned thread is still running, reports the pod as not live so the platform
restarts it and the worker slot comes back. This mirrors what the Ray engine
does for abandoned conversions.
"""

import asyncio
import datetime
import logging
from dataclasses import dataclass
from typing import Any

from docling_jobkit.datamodel.task import Task
from docling_jobkit.datamodel.task_meta import TaskStatus

_log = logging.getLogger(__name__)

STUCK_TASK_ERROR_MESSAGE = (
    "The conversion did not finish within its deadline and was abandoned."
)
STUCK_TASK_HEALTH_DETAIL = (
    "A conversion thread is stuck past its deadline; restart to free the worker."
)


def task_deadline_seconds(task: Task, grace_seconds: float) -> float | None:
    """Seconds a started task may run before it counts as stuck.

    ``document_timeout`` applies to each document, so a task gets one timeout
    per source plus the grace period. Returns None when the task has no
    document timeout (nothing to enforce).
    """
    options = task.convert_options
    timeout = getattr(options, "document_timeout", None) if options else None
    if timeout is None or timeout <= 0:
        return None
    num_sources = max(1, len(task.sources))
    return float(timeout) * num_sources + grace_seconds


@dataclass
class _AbandonedTask:
    task: Task
    abandoned_at: datetime.datetime


class LocalStuckTaskWatchdog:
    """Periodically fail started tasks that are past their deadline.

    ``orchestrator`` is a ``LocalOrchestrator``; only its ``tasks`` dict,
    ``notifier`` and ``_task_results`` are used.
    """

    def __init__(
        self,
        orchestrator: Any,
        grace_seconds: float,
        check_interval_seconds: float,
    ):
        self.orchestrator = orchestrator
        self.grace_seconds = grace_seconds
        self.check_interval_seconds = check_interval_seconds
        self._abandoned: dict[str, _AbandonedTask] = {}

    @property
    def stuck_task_ids(self) -> list[str]:
        """Tasks that were failed by the watchdog and whose thread still runs."""
        return list(self._abandoned)

    def has_stuck_tasks(self) -> bool:
        return bool(self._abandoned)

    async def check(self, now: datetime.datetime | None = None) -> list[Task]:
        """Run one scan. Returns the tasks newly failed by this scan."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        await self._release_finished_threads()

        newly_failed: list[Task] = []
        for task in list(self.orchestrator.tasks.values()):
            if task.task_status != TaskStatus.STARTED or task.started_at is None:
                continue
            if task.task_id in self._abandoned:
                continue
            deadline = task_deadline_seconds(task, self.grace_seconds)
            if deadline is None:
                continue
            elapsed = (now - task.started_at).total_seconds()
            if elapsed <= deadline:
                continue

            _log.error(
                "Task %s has been running for %.0fs, past its %.0fs deadline "
                "(document_timeout x %d source(s) + %.0fs grace). Marking it "
                "failed; its worker thread cannot be stopped and stays busy.",
                task.task_id,
                elapsed,
                deadline,
                max(1, len(task.sources)),
                self.grace_seconds,
            )
            task.error_message = STUCK_TASK_ERROR_MESSAGE
            task.set_status(TaskStatus.FAILURE)
            self._abandoned[task.task_id] = _AbandonedTask(
                task=task, abandoned_at=task.last_update_at
            )
            await self._notify(task.task_id)
            newly_failed.append(task)
        return newly_failed

    async def _release_finished_threads(self) -> None:
        """Forget abandoned tasks whose worker thread has returned.

        The local worker updates the task status when its thread returns, which
        moves ``last_update_at`` past the moment the watchdog failed it. The
        client was already told the task failed, so a late result is dropped
        and the failure is kept.
        """
        for task_id, entry in list(self._abandoned.items()):
            task = entry.task
            if task.last_update_at <= entry.abandoned_at:
                continue  # thread still running
            del self._abandoned[task_id]
            _log.warning(
                "The worker thread of abandoned task %s returned (status %s); "
                "its worker slot is free again.",
                task_id,
                task.task_status.value,
            )
            changed = task.error_message != STUCK_TASK_ERROR_MESSAGE
            task.error_message = STUCK_TASK_ERROR_MESSAGE
            if task.task_status != TaskStatus.FAILURE:
                results = getattr(self.orchestrator, "_task_results", None)
                if isinstance(results, dict):
                    results.pop(task_id, None)
                task.set_status(TaskStatus.FAILURE)
                changed = True
            if changed and task_id in self.orchestrator.tasks:
                await self._notify(task_id)

    async def _notify(self, task_id: str) -> None:
        notifier = getattr(self.orchestrator, "notifier", None)
        if notifier is None:
            return
        try:
            await notifier.notify_task_subscribers(task_id)
        except Exception:
            _log.exception("Could not notify subscribers of stuck task %s", task_id)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.check_interval_seconds)
            try:
                await self.check()
            except Exception:
                _log.exception("Stuck-task watchdog scan failed")
