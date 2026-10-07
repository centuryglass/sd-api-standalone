"""Async generation-handle integration tests for the WebUI backend.

These exercise ``A1111Webservice.submit_txt2img`` / ``submit_img2img`` and the client-side dispatch
queue (:class:`WebUIDispatcher` / :class:`WebUIGenerationHandle`) against a **real** WebUI server. They
run actual diffusion, so they are opt-in like the other generation tests (``--run-generation`` /
``RUN_SD_GENERATION=1``) and need a checkpoint loaded.

What is covered:
- the full async lifecycle (PENDING -> ACTIVE -> FINISHED) and progress callbacks;
- ``submit_upscale`` and ``submit_preprocessor_preview`` returning their one image through a handle;
- client-side single-slot serialization of two concurrent submissions;
- clean cancellation of a still-PENDING job vs. interrupt-based cancellation of an ACTIVE one;
- ``_await_server_idle``: a submission is held PENDING (not dispatched) while the server is busy with
  an *external* client's job, and dispatches once the server drains;
- cancelling a job held for a busy server drops it cleanly without interrupting the external job.

The "server busy" tests deliberately scale an external job's step count up so it occupies the server long
enough to observe deferral; if the hardware is fast enough to finish it before we can look, the test
fails with a clear "raise the step count" message rather than passing vacuously.
"""
import threading
import time

import pytest
from PIL import Image

from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.shared_data.generation_handle import GenerationError, GenerationStatus
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody

from .helpers import make_edge_image, make_structured_image, save_output

pytestmark = [pytest.mark.integration, pytest.mark.generation]


def _body(prompt: str = 'a red apple on a wooden table', *, steps: int = 20, size: int = 512,
          seed: int = 1) -> DiffusionRequestBody:
    """A self-contained request body; step count/size are tuned per-test for timing control."""
    body = DiffusionRequestBody()
    body.prompt = prompt
    body.negative_prompt = ''
    body.steps = steps
    body.cfg_scale = 4.0
    body.width = size
    body.height = size
    body.batch_size = 1
    body.n_iter = 1
    body.seed = seed
    body.save_images = False
    body.send_images = True
    return body


def _assert_valid_image(image, expected_size: int) -> None:
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGBA'  # decoded images are normalized to RGBA
    assert abs(image.width - expected_size) <= 8
    assert abs(image.height - expected_size) <= 8


class _StatusPoller:
    """Background thread that samples a handle's status until it terminates, recording every snapshot."""

    def __init__(self, handle, interval: float = 0.05):
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


def _wait_until(predicate, timeout: float, interval: float = 0.05, message: str = 'condition') -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f'Timed out after {timeout}s waiting for {message}')


# --------------------------------------------------------------------------- #
# Basic lifecycle
# --------------------------------------------------------------------------- #

def test_submit_txt2img_lifecycle(service, output_dir):
    """submit -> wait yields an image, and the ACTIVE state is observed via the progress callback."""
    handle = service.submit_txt2img(_body('a red apple on a wooden table'))
    assert handle.task_id and handle.task_id.startswith('task(txt2img-')
    assert handle.poll().status is GenerationStatus.PENDING  # not dispatched synchronously

    seen: list[GenerationStatus] = []
    result = handle.wait(timeout=120, poll_interval=0.2, on_progress=lambda p: seen.append(p.status))

    assert len(result.images) == 1
    _assert_valid_image(result.images[0], expected_size=512)
    save_output(output_dir, 'async_txt2img', result.images[0])
    assert result.task_id == handle.task_id
    assert GenerationStatus.ACTIVE in seen, f'never observed ACTIVE; saw {[s.value for s in seen]}'
    assert seen[-1] is GenerationStatus.FINISHED


def test_submit_img2img_lifecycle(service, output_dir):
    """The img2img submit path round-trips a source image through the async handle."""
    source = Image.new('RGBA', (512, 512), (40, 90, 160, 255))
    body = _body('turn this into an oil painting', seed=7)
    body.denoising_strength = 0.75
    body.init_images = [source]
    handle = service.submit_img2img(body)
    result = handle.wait(timeout=120)
    assert len(result.images) >= 1
    _assert_valid_image(result.images[0], expected_size=512)
    save_output(output_dir, 'async_img2img', result.images[0])


def test_submit_upscale_lifecycle(service, output_dir):
    """A basic upscale runs through the dispatcher and yields one image at the requested size."""
    handle = service.submit_upscale(make_structured_image(128, 128), 256, 256)
    assert handle.task_id and handle.task_id.startswith('task(upscale-')
    result = handle.wait(timeout=120)
    assert len(result.images) == 1
    _assert_valid_image(result.images[0], expected_size=256)
    save_output(output_dir, 'async_upscale', result.images[0])


@pytest.mark.controlnet
def test_submit_preprocessor_preview_lifecycle(controlnet_available, controlnet_pairing, output_dir):
    """A preprocessor preview runs through the dispatcher and yields one non-flat control image."""
    module, _model = controlnet_pairing
    preprocessor = next((p for p in controlnet_available.get_controlnet_preprocessors() if p.name == module), None)
    if preprocessor is None:
        pytest.skip(f'Preprocessor {module!r} not found among available modules.')
    result = controlnet_available.submit_preprocessor_preview(make_edge_image(), preprocessor).wait(timeout=120)
    assert len(result.images) == 1
    save_output(output_dir, f'async_controlnet_preview_{module}', result.images[0])
    extrema = result.images[0].convert('L').getextrema()
    assert extrema[0] != extrema[1], 'preprocessor preview is a flat image; preprocessing likely did nothing'


# --------------------------------------------------------------------------- #
# Client-side serialization
# --------------------------------------------------------------------------- #

def test_concurrent_submits_serialize_single_slot(service, output_dir):
    """Two submissions never run at once: one is ACTIVE while the other stays PENDING (queued client-side)."""
    h1 = service.submit_txt2img(_body('a blue teapot', steps=25, seed=2))
    h2 = service.submit_txt2img(_body('a green frog', steps=25, seed=3))

    with _StatusPoller(h1) as p1, _StatusPoller(h2) as p2:
        r1 = h1.wait(timeout=120)
        r2 = h2.wait(timeout=120)

    assert r1.images and r2.images
    save_output(output_dir, 'async_serialize_1', r1.images[0])
    save_output(output_dir, 'async_serialize_2', r2.images[0])

    # The second-dispatched job must have been observed PENDING (held client-side) at some point, and a
    # PENDING snapshot must have reported queue_index 0 (waiting directly behind the running job).
    all_snaps = p1.snapshots + p2.snapshots
    pending_indices = [s.queue_index for s in all_snaps
                       if s.status is GenerationStatus.PENDING and s.queue_index is not None]
    assert GenerationStatus.PENDING in (p1.statuses() | p2.statuses()), 'jobs were not serialized'
    assert 0 in pending_indices, f'no PENDING job reported queue_index 0; saw {pending_indices}'


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #

def test_cancel_pending_job_is_clean(service):
    """Cancelling a still-PENDING job removes it client-side: CANCELLED, no image, no server work."""
    ha = service.submit_txt2img(_body('a yellow lemon', steps=25, seed=4))
    hb = service.submit_txt2img(_body('a purple grape', steps=25, seed=5))

    _wait_until(lambda: ha.poll().status is GenerationStatus.ACTIVE, timeout=30, message='first job ACTIVE')
    assert hb.poll().status is GenerationStatus.PENDING

    assert hb.cancel() is True
    assert hb.poll().status is GenerationStatus.CANCELLED

    with pytest.raises(GenerationError):
        hb.wait(timeout=5)
    # The un-cancelled job still completes normally.
    assert ha.wait(timeout=120).images


def test_cancel_active_job_interrupts(service):
    """Cancelling the ACTIVE job interrupts it server-side and the handle ends CANCELLED."""
    handle = service.submit_txt2img(_body('an intricate cathedral, highly detailed', steps=60, seed=6))
    _wait_until(lambda: handle.poll().status is GenerationStatus.ACTIVE, timeout=30, message='job ACTIVE')
    time.sleep(0.5)  # let a few steps run so the interrupt lands mid-generation

    assert handle.cancel() is True
    _wait_until(lambda: handle.poll().status.is_terminal, timeout=120, message='job terminal')
    assert handle.poll().status is GenerationStatus.CANCELLED
    with pytest.raises(GenerationError):
        handle.wait(timeout=5)


# --------------------------------------------------------------------------- #
# _await_server_idle: hold dispatch while another client occupies the server
# --------------------------------------------------------------------------- #

def _occupy_server(api_url, credentials, *, steps: int, size: int, seed: int):
    """Start a slow blocking job from an independent client; return (thread, result_dict, service)."""
    external = A1111Webservice(api_url, credentials_provider=lambda: credentials)
    external.get_samplers()  # drive login before timing matters
    result: dict = {}

    def run():
        result['response'] = external.txt2img(_body('a vast, intricate fantasy landscape',
                                                     steps=steps, size=size, seed=seed))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result, external


def test_dispatch_deferred_while_server_busy(service, api_url, credentials, output_dir):
    """A submission is held PENDING (not dispatched) while an external client's job occupies the server."""
    ext_thread, ext_result, external = _occupy_server(api_url, credentials, steps=80, size=768, seed=101)
    try:
        # Confirm the external job is actually running on the server before we submit ours.
        _wait_until(lambda: service.progress_check().state.job_count > 0, timeout=30,
                    message='external job to occupy the server')

        handle = service.submit_txt2img(_body('a small red cube', steps=8, seed=102))
        time.sleep(1.5)  # ample time for the worker to dispatch, if it weren't waiting for idle

        assert ext_thread.is_alive(), \
            'external job finished before we could observe deferral — raise its step count'
        assert handle.poll().status is GenerationStatus.PENDING, \
            'job was dispatched while the server was still busy (idle-wait not holding it)'
    finally:
        ext_thread.join(timeout=180)

    assert ext_result['response']['images'], 'external job did not complete'
    # Once the server drained, our held job should dispatch and finish on its own.
    result = handle.wait(timeout=120)
    assert result.images
    save_output(output_dir, 'async_idle_wait', result.images[0])
    external.disconnect()


def test_cancel_while_held_for_busy_server_is_clean(service, api_url, credentials):
    """A job cancelled while held for a busy server drops cleanly and never interrupts the external job."""
    ext_thread, ext_result, external = _occupy_server(api_url, credentials, steps=80, size=768, seed=201)
    try:
        _wait_until(lambda: service.progress_check().state.job_count > 0, timeout=30,
                    message='external job to occupy the server')

        handle = service.submit_txt2img(_body('a small blue cube', steps=8, seed=202))
        time.sleep(1.0)
        assert ext_thread.is_alive(), 'external job finished too fast — raise its step count'
        assert handle.poll().status is GenerationStatus.PENDING

        # Cancel while held: must be clean (no POST was ever sent, so nothing to interrupt).
        assert handle.cancel() is True
        _wait_until(lambda: handle.poll().status.is_terminal, timeout=10, message='cancelled job terminal')
        assert handle.poll().status is GenerationStatus.CANCELLED
    finally:
        ext_thread.join(timeout=180)

    # The external job must have run to completion undisturbed (our cancel didn't interrupt it).
    assert ext_result['response']['images'], 'external job was disrupted by our cancel'
    external.disconnect()
