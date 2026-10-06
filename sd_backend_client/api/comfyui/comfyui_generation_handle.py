"""ComfyUI implementation of the backend-agnostic generation handle.

ComfyUI is natively asynchronous: submitting a workflow to ``/prompt`` returns immediately with a
``prompt_id`` (and a queue ``number``), and you poll ``/history`` + ``/queue`` — wrapped here by
``ComfyUiWebservice.check_queue_entry`` — until the job finishes, then download the resulting image
references. That maps almost one-to-one onto :class:`GenerationHandle`:

- ``poll()``          -> ``check_queue_entry(prompt_id, number)``, translating ``AsyncTaskStatus``.
- ``_build_result()`` -> ``download_images`` on the finished job's output references.
- ``cancel()``        -> drop a still-queued job from the queue, or interrupt the running one.

Because ComfyUI is already async there is no client-side dispatch queue here (unlike the WebUI handle);
the server *is* the queue. A handle just tracks one ``prompt_id`` and reports what the server says about
it, with two small adaptations layered on top of the raw server status:

* **Cancellation is inferred, not reported.** ComfyUI has no "cancelled" task state — a removed queue
  entry simply disappears (``NOT_FOUND``) and an interrupted run leaves an incomplete/absent history
  entry (``NOT_FOUND`` / ``FAILED``). So once ``cancel()`` has been requested, those terminal states are
  reported as ``CANCELLED`` (mirroring the WebUI handle's "cancel wins" behavior).
* **Post-submit registration lag is smoothed.** For a brief moment right after ``/prompt`` returns, the
  job may not yet be visible in the queue, which would read as ``NOT_FOUND`` (a terminal state). Until
  the job has been seen at least once, a ``NOT_FOUND`` is reported as ``PENDING`` instead, so a fast
  ``poll()``/``wait()`` immediately after submission doesn't spuriously fail.

No Qt / numpy / cv2 here: results are plain ``PIL.Image`` (per the repo's hard constraints).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional, TYPE_CHECKING

from sd_backend_client.api.comfyui_webservice import AsyncTaskStatus
from sd_backend_client.api.shared_data.generation_handle import (GenerationHandle, GenerationProgress,
                                                              GenerationResult, GenerationStatus)
from sd_backend_client.errors import WorkflowValidationError

if TYPE_CHECKING:
    from sd_backend_client.api.comfyui.comfyui_types import PromptExecOutputs, QueueAdditionResponse
    from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice

logger = logging.getLogger(__name__)

# ComfyUI's queue-status enum maps onto the unified lifecycle almost verbatim; it has no CANCELLED (see
# the module docstring — cancellation is inferred from NOT_FOUND / FAILED after a cancel is requested).
_STATUS_MAP: dict[AsyncTaskStatus, GenerationStatus] = {
    AsyncTaskStatus.PENDING: GenerationStatus.PENDING,
    AsyncTaskStatus.ACTIVE: GenerationStatus.ACTIVE,
    AsyncTaskStatus.FINISHED: GenerationStatus.FINISHED,
    AsyncTaskStatus.FAILED: GenerationStatus.FAILED,
    AsyncTaskStatus.NOT_FOUND: GenerationStatus.NOT_FOUND,
}


class ComfyGenerationHandle(GenerationHandle):
    """Handle to one ComfyUI ``/prompt`` job, identified by its ``prompt_id`` (+ queue ``number``)."""

    def __init__(self, service: 'ComfyUiWebservice', prompt_id: str, number: int,
                 seed: Optional[int] = None) -> None:
        super().__init__(prompt_id)
        self._service = service
        self._number = number
        self._seed = seed
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._ever_registered = False           # Has the server ever reported this job as real?
        self._outputs: Optional['PromptExecOutputs'] = None  # Cached from the FINISHED poll.

    @classmethod
    def from_queue_response(cls, service: 'ComfyUiWebservice',
                            response: 'QueueAdditionResponse') -> 'ComfyGenerationHandle':
        """Build a handle from a ``txt2img`` / ``img2img`` / ``inpaint`` queue response.

        Raises ``WorkflowValidationError`` if the response has no ``prompt_id`` or queue ``number``, which means the
        server rejected the workflow.
        """
        if response.prompt_id is None or response.number is None:
            raise WorkflowValidationError(response.error, response.node_errors)
        return cls(service, response.prompt_id, response.number, seed=response.seed)

    # --- GenerationHandle surface -----------------------------------------------------------------

    def poll(self) -> GenerationProgress:
        assert self._task_id is not None  # guaranteed by from_queue_response
        progress = self._service.check_queue_entry(self._task_id, self._number)
        status = _STATUS_MAP[progress.status]

        with self._lock:
            if progress.outputs is not None:
                self._outputs = progress.outputs
            if status in (GenerationStatus.PENDING, GenerationStatus.ACTIVE, GenerationStatus.FINISHED):
                self._ever_registered = True
            cancel_requested = self._cancel_requested
            ever_registered = self._ever_registered

        # A removed/interrupted job disappears or errors out; honor the caller's cancel intent.
        if cancel_requested and status in (GenerationStatus.NOT_FOUND, GenerationStatus.FAILED):
            return GenerationProgress(GenerationStatus.CANCELLED)
        # Smooth over the brief window before the freshly-queued job shows up in the queue.
        if status is GenerationStatus.NOT_FOUND and not ever_registered:
            return GenerationProgress(GenerationStatus.PENDING, text_info='Queued (awaiting registration)')

        queue_index = progress.index if status is GenerationStatus.PENDING else None
        return GenerationProgress(status, queue_index=queue_index)

    def _build_result(self) -> GenerationResult:
        with self._lock:
            outputs = self._outputs
        if outputs is None:
            # Only reached if _build_result is called without a preceding FINISHED poll; fetch fresh.
            assert self._task_id is not None
            outputs = self._service.check_queue_entry(self._task_id, self._number).outputs
        image_refs = outputs.images if (outputs is not None and outputs.images is not None) else []
        images = self._service.download_images(image_refs)
        return GenerationResult(images=images, info=outputs, seed=self._seed, task_id=self._task_id)

    def cancel(self) -> bool:
        """Cancel the job: drop it from the queue if still PENDING, else interrupt the running job.

        Returns ``True`` if a cancellation was issued, ``False`` if the job had already finished. Unlike a
        blanket ``interrupt(task_id)``, cancelling a *pending* job here leaves any unrelated running job
        untouched (see ``ComfyUiWebservice.remove_from_queue``).
        """
        assert self._task_id is not None
        status = _STATUS_MAP[self._service.check_queue_entry(self._task_id, self._number).status]
        if status is GenerationStatus.FINISHED:
            return False
        with self._lock:
            self._cancel_requested = True

        try:
            if status is GenerationStatus.PENDING:
                self._service.remove_from_queue(self._task_id)   # clean: doesn't touch the running job
            else:
                self._service.interrupt()                        # ACTIVE (or settling): stop the run
        except Exception:  # pragma: no cover - cancellation is best-effort
            logger.exception('Failed to cancel ComfyUI job %s', self._task_id)
        return True
