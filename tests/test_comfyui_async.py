"""Async generation-handle integration tests for the ComfyUI backend.

These exercise ``ComfyUiWebservice.submit_txt2img`` / ``submit_img2img`` and :class:`ComfyGenerationHandle`
against a **real** ComfyUI server. ComfyUI is natively async (the server owns the queue), so unlike the
WebUI there is no client-side dispatcher — the handle just tracks one ``prompt_id`` and reports the
server's view of it through the unified :class:`GenerationHandle` interface.

Covered: the submit -> wait lifecycle, server-side queueing of two jobs, interrupt-based cancellation of
an ACTIVE job, and — the key check for our ``remove_from_queue`` path — cancelling a still-PENDING job
without disturbing the job currently running.

Opt-in like the other generation tests (``--run-generation`` / ``RUN_SD_GENERATION=1``); needs a
checkpoint installed. The cancellation tests scale a job up (steps/size) so it stays running long enough
to observe, and fail with a clear "raise the step count" message if the hardware finishes it too fast.
"""
import threading
import time

import pytest
from PIL import Image

from sd_backend_client.api.shared_data.generation_handle import GenerationError, GenerationStatus

from .comfy_helpers import build_comfy_params
from .helpers import save_output

pytestmark = [pytest.mark.integration, pytest.mark.generation]


def _params(comfy_checkpoint, *, prompt='a red apple on a wooden table', steps=8, size=256, seed=1):
    """A ComfyUI param set with per-test control over step count / size (for timing)."""
    params = build_comfy_params(comfy_checkpoint, prompt=prompt)
    params.steps = steps
    params.width = size
    params.height = size
    params.seed = seed
    return params


def _assert_valid_image(image, expected_size):
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGBA'
    assert abs(image.width - expected_size) <= 8
    assert abs(image.height - expected_size) <= 8


def _wait_until(predicate, timeout, interval=0.05, message='condition'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f'Timed out after {timeout}s waiting for {message}')


class _StatusPoller:
    """Background thread sampling a handle's status until it terminates, recording every snapshot."""

    def __init__(self, handle, interval=0.05):
        self.handle = handle
        self.interval = interval
        self.snapshots = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            snap = self.handle.poll()
            self.snapshots.append(snap)
            if snap.status.is_terminal:
                break
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def statuses(self):
        return {snap.status for snap in self.snapshots}


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

def test_submit_txt2img_lifecycle(comfy_service, comfy_checkpoint, output_dir):
    """submit returns immediately (non-terminal), then wait() drives it to a downloadable image."""
    handle = comfy_service.submit_txt2img(_params(comfy_checkpoint))
    assert handle.task_id
    assert not handle.poll().status.is_terminal, 'submit should not block until the job finishes'

    seen = []
    result = handle.wait(timeout=180, poll_interval=0.2, on_progress=lambda p: seen.append(p.status))

    assert len(result.images) >= 1
    _assert_valid_image(result.images[0], expected_size=256)
    save_output(output_dir, 'comfy_async_txt2img', result.images[0])
    assert result.task_id == handle.task_id
    assert seen[-1] is GenerationStatus.FINISHED


def test_submit_img2img_lifecycle(comfy_service, comfy_checkpoint, output_dir):
    """The img2img submit path round-trips a source image through the async handle."""
    params = _params(comfy_checkpoint, prompt='turn this into an oil painting', seed=7)
    params.denoising_strength = 0.75
    params.init_images = [Image.new('RGBA', (256, 256), (40, 90, 160, 255))]
    handle = comfy_service.submit_img2img(params)
    result = handle.wait(timeout=180)
    assert len(result.images) >= 1
    _assert_valid_image(result.images[0], expected_size=256)
    save_output(output_dir, 'comfy_async_img2img', result.images[0])


# --------------------------------------------------------------------------- #
# Server-side queueing
# --------------------------------------------------------------------------- #

def test_two_jobs_queue_on_server(comfy_service, comfy_checkpoint, output_dir):
    """Two submissions both complete; the second is observed PENDING (queued behind the first)."""
    h1 = comfy_service.submit_txt2img(_params(comfy_checkpoint, prompt='a blue teapot', steps=25, seed=2))
    h2 = comfy_service.submit_txt2img(_params(comfy_checkpoint, prompt='a green frog', steps=25, seed=3))

    with _StatusPoller(h1) as p1, _StatusPoller(h2) as p2:
        r1 = h1.wait(timeout=180)
        r2 = h2.wait(timeout=180)

    assert r1.images and r2.images
    save_output(output_dir, 'comfy_async_queue_1', r1.images[0])
    save_output(output_dir, 'comfy_async_queue_2', r2.images[0])
    assert GenerationStatus.PENDING in (p1.statuses() | p2.statuses()), \
        'neither job was ever observed queued (PENDING) — jobs did not serialize on the server'


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #

def test_cancel_active_job_interrupts(comfy_service, comfy_checkpoint):
    """Cancelling the running job interrupts it server-side and the handle ends CANCELLED."""
    handle = comfy_service.submit_txt2img(
        _params(comfy_checkpoint, prompt='an intricate cathedral, highly detailed', steps=60, size=512, seed=6))
    _wait_until(lambda: handle.poll().status is GenerationStatus.ACTIVE, timeout=60, message='job ACTIVE')
    time.sleep(0.5)  # let a few steps run so the interrupt lands mid-generation

    assert handle.cancel() is True
    _wait_until(lambda: handle.poll().status.is_terminal, timeout=120, message='job terminal')
    assert handle.poll().status is GenerationStatus.CANCELLED
    with pytest.raises(GenerationError):
        handle.wait(timeout=5)


def test_cancel_pending_job_leaves_running_job_untouched(comfy_service, comfy_checkpoint, output_dir):
    """Cancelling a queued job drops it cleanly via remove_from_queue without interrupting the running one."""
    # A long first job occupies the server; a second job then waits behind it, PENDING.
    running = comfy_service.submit_txt2img(
        _params(comfy_checkpoint, prompt='a vast, intricate fantasy landscape', steps=80, size=512, seed=101))
    queued = comfy_service.submit_txt2img(_params(comfy_checkpoint, prompt='a small red cube', steps=8, seed=102))

    _wait_until(lambda: running.poll().status is GenerationStatus.ACTIVE, timeout=60, message='first job ACTIVE')
    assert queued.poll().status is GenerationStatus.PENDING, \
        'second job was not PENDING — the first finished too fast; raise its step/size'

    assert queued.cancel() is True
    _wait_until(lambda: queued.poll().status.is_terminal, timeout=30, message='queued job terminal')
    assert queued.poll().status is GenerationStatus.CANCELLED

    # The running job must complete undisturbed — remove_from_queue must not have interrupted it.
    result = running.wait(timeout=180)
    assert result.images, 'the running job was disrupted by cancelling the queued one'
    save_output(output_dir, 'comfy_async_running_survived_cancel', result.images[0])
