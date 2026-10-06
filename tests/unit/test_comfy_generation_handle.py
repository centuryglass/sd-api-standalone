"""Offline unit tests for :class:`ComfyGenerationHandle`.

These pin the handle's server-status translation and cancellation logic without a live ComfyUI server,
using a scripted fake service. They cover the pieces that are easy to get subtly wrong:

- ``AsyncTaskStatus`` -> ``GenerationStatus`` mapping and PENDING queue-index passthrough;
- the full PENDING -> ACTIVE -> FINISHED ``wait()`` path, including result download;
- "cancel wins": once cancelled, a NOT_FOUND / FAILED server state reports CANCELLED;
- clean PENDING cancel (queue removal) vs. ACTIVE cancel (interrupt), and no-op cancel of a finished job;
- registration-lag smoothing: a NOT_FOUND seen before the job is ever registered reads as PENDING.
"""
from PIL import Image

import pytest

from sd_backend_client.api.comfyui.comfyui_generation_handle import ComfyGenerationHandle
from sd_backend_client.api.comfyui.comfyui_types import (ImageFileReference, PromptExecOutputs,
                                                      QueueAdditionResponse)
from sd_backend_client.api.comfyui_webservice import AsyncTaskProgress, AsyncTaskStatus
from sd_backend_client.api.shared_data.generation_handle import (GenerationError, GenerationStatus)
from sd_backend_client.errors import BackendTimeoutError, WorkflowValidationError


class FakeComfyService:
    """A stand-in for ComfyUiWebservice that replays a scripted sequence of task-progress snapshots."""

    def __init__(self, script: list[AsyncTaskProgress]):
        self._script = list(script)
        self._calls = 0
        self.removed: list[str] = []       # task_ids passed to remove_from_queue
        self.interrupted: list = []         # task_ids (or None) passed to interrupt

    def check_queue_entry(self, prompt_id: str, number) -> AsyncTaskProgress:
        # Advance through the script, clamping on the last snapshot for any further polls.
        snapshot = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return snapshot

    def remove_from_queue(self, task_id: str) -> None:
        self.removed.append(task_id)

    def interrupt(self, task_id=None) -> None:
        self.interrupted.append(task_id)

    def download_images(self, image_refs):
        return [Image.new('RGBA', (4, 4), (i, i, i, 255)) for i, _ in enumerate(image_refs)]


def _outputs(n: int = 1) -> PromptExecOutputs:
    return PromptExecOutputs(images=[ImageFileReference(filename=f'out_{i}.png', subfolder='')
                                     for i in range(n)])


def _handle(script, seed=None):
    return ComfyGenerationHandle(FakeComfyService(script), 'prompt-123', number=7, seed=seed)


# --------------------------------------------------------------------------- #
# from_queue_response
# --------------------------------------------------------------------------- #

def test_from_queue_response_populates_ids_and_seed():
    response = QueueAdditionResponse(prompt_id='abc', number=4, seed=99)
    handle = ComfyGenerationHandle.from_queue_response(FakeComfyService([]), response)
    assert handle.task_id == 'abc'
    assert handle._number == 4
    assert handle._seed == 99


def test_from_queue_response_rejects_missing_prompt_id():
    response = QueueAdditionResponse(prompt_id=None, error='node validation failed')
    with pytest.raises(WorkflowValidationError, match='rejected the workflow: node validation failed'):
        ComfyGenerationHandle.from_queue_response(FakeComfyService([]), response)


def test_from_queue_response_rejects_missing_number():
    """A response without a queue number is rejected, since check_queue_entry needs it to find a pending job."""
    response = QueueAdditionResponse(prompt_id='abc', number=None)
    with pytest.raises(WorkflowValidationError, match='rejected the workflow'):
        ComfyGenerationHandle.from_queue_response(FakeComfyService([]), response)


# --------------------------------------------------------------------------- #
# Status mapping
# --------------------------------------------------------------------------- #

def test_poll_maps_pending_with_queue_index():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=3)])
    progress = handle.poll()
    assert progress.status is GenerationStatus.PENDING
    assert progress.queue_index == 3


def test_poll_maps_active_and_clears_queue_index():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE, index=3)])
    progress = handle.poll()
    assert progress.status is GenerationStatus.ACTIVE
    assert progress.queue_index is None  # index only meaningful while PENDING


def test_poll_maps_failed():
    # Register the job first (ACTIVE) so the later FAILED isn't smoothed as a registration-lag case.
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),
                      AsyncTaskProgress(status=AsyncTaskStatus.FAILED)])
    handle.poll()
    assert handle.poll().status is GenerationStatus.FAILED


# --------------------------------------------------------------------------- #
# Lifecycle / wait
# --------------------------------------------------------------------------- #

def test_wait_runs_through_lifecycle_and_downloads(tmp_path):
    handle = _handle([
        AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=1),
        AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),
        AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs(2)),
    ], seed=42)
    seen = []
    result = handle.wait(poll_interval=0.0, on_progress=lambda p: seen.append(p.status))
    assert [s for s in (GenerationStatus.PENDING, GenerationStatus.ACTIVE, GenerationStatus.FINISHED)
            if s in seen] == [GenerationStatus.PENDING, GenerationStatus.ACTIVE, GenerationStatus.FINISHED]
    assert len(result.images) == 2
    assert result.seed == 42
    assert result.task_id == 'prompt-123'
    assert isinstance(result.info, PromptExecOutputs)


def test_build_result_uses_cached_outputs_from_finished_poll():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs(1))])
    handle.poll()  # caches outputs
    result = handle._build_result()
    assert len(result.images) == 1


# --------------------------------------------------------------------------- #
# Registration-lag smoothing
# --------------------------------------------------------------------------- #

def test_not_found_before_registration_reads_as_pending():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])
    progress = handle.poll()
    assert progress.status is GenerationStatus.PENDING  # smoothed: job not yet visible in the queue


def test_not_found_after_registration_is_terminal():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),
                      AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])
    assert handle.poll().status is GenerationStatus.ACTIVE  # registers the job
    assert handle.poll().status is GenerationStatus.NOT_FOUND


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #

def test_cancel_pending_removes_from_queue_without_interrupt():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=2),  # cancel() reads this
                      AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])         # subsequent poll
    assert handle.cancel() is True
    assert handle._service.removed == ['prompt-123']
    assert handle._service.interrupted == []           # a queued job must not interrupt the running one
    assert handle.poll().status is GenerationStatus.CANCELLED  # cancel wins over NOT_FOUND


def test_cancel_active_interrupts():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),   # cancel() reads this
                      AsyncTaskProgress(status=AsyncTaskStatus.FAILED)])  # interrupted run errors out
    assert handle.cancel() is True
    assert handle._service.interrupted == [None]       # interrupt the running job (no task_id)
    assert handle._service.removed == []
    assert handle.poll().status is GenerationStatus.CANCELLED  # cancel wins over FAILED


def test_cancel_finished_job_is_noop():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs(1))])
    assert handle.cancel() is False
    assert handle._service.removed == []
    assert handle._service.interrupted == []


def test_cancelled_job_wait_raises():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=0),
                      AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])
    handle.cancel()
    with pytest.raises(GenerationError):
        handle.wait(poll_interval=0.0)


def test_wait_timeout_raises_backend_timeout_error():
    """The polling wait() raises BackendTimeoutError, which is also a TimeoutError, when the timeout elapses."""
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE)] * 3)
    with pytest.raises(BackendTimeoutError) as error:
        handle.wait(timeout=0, poll_interval=0.0)
    assert isinstance(error.value, TimeoutError)
