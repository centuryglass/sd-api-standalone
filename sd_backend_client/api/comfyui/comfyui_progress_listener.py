"""Live per-job progress for ComfyUI jobs, read from the server's websocket on one background thread.

ComfyUI reports step progress and preview frames only over its websocket, and only to the connection whose
`clientId` submitted the job. The server keeps one connection per `clientId`, so one listener serves every job a
`ComfyUiWebservice` submits; a second connection with the same id would take the messages away from the first.

The listener is best-effort. If the websocket cannot be opened, jobs still run and poll normally, just without
progress, and the listener retries no sooner than `retry_interval` seconds later.
"""
from __future__ import annotations

import json
import logging
import struct
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import websocket
from PIL import Image

from sd_backend_client.errors import SDBackendError
from sd_backend_client.util.visual.image_utils import image_from_bytes

logger = logging.getLogger(__name__)

DEFAULT_RETRY_INTERVAL = 5.0
DEFAULT_RECEIVE_TIMEOUT = 1.0

# Binary websocket frames start with a big-endian uint32 event type (comfy_execution BinaryEventTypes).
_PREVIEW_IMAGE = 1
_PREVIEW_IMAGE_WITH_METADATA = 4
_UINT32 = struct.Struct('>I')

# Text messages sent once a job has ended. ComfyUI also sends an 'executing' message with no node when a job ends.
_END_MESSAGES = ('execution_success', 'execution_error', 'execution_interrupted')


@dataclass
class LiveProgress:
    """The latest websocket-reported state of one job. Every field is None until the server reports it."""
    progress: Optional[float] = None       # Fraction complete in [0, 1] for the node currently reporting steps.
    eta_seconds: Optional[float] = None    # Estimated seconds left in that node, from its step rate so far.
    preview: Optional[Image.Image] = None  # Latest preview frame, sent only when ComfyUI runs with a preview method.


@dataclass
class _JobState:
    live: LiveProgress
    node: Optional[str] = None
    first_step: Optional[tuple[float, float]] = None  # (time, value) of the node's first progress message.


class ComfyProgressListener:
    """Tracks progress for watched ComfyUI jobs from one websocket connection.

    `watch` starts the background thread if it is not running, and the thread exits once no job is watched. Each
    `ComfyGenerationHandle` watches its job while it is unfinished and unwatches it when it ends. The listener also
    drops a job when the server reports it ended, so a handle that is never polled again doesn't keep the thread
    alive.
    """

    def __init__(self, open_websocket: Callable[[], AbstractContextManager[Any]],
                 clock: Callable[[], float] = time.monotonic,
                 retry_interval: float = DEFAULT_RETRY_INTERVAL,
                 receive_timeout: float = DEFAULT_RECEIVE_TIMEOUT) -> None:
        """
        Parameters
        ----------
        open_websocket: Callable[[], AbstractContextManager[Any]]
            Opens the server websocket, as `ComfyUiWebservice.open_websocket` does.
        clock: Callable[[], float], default=time.monotonic
            Time source for ETA estimates and connection retries.
        retry_interval: float, default=DEFAULT_RETRY_INTERVAL
            Minimum seconds between a failed connection and the next attempt.
        receive_timeout: float, default=DEFAULT_RECEIVE_TIMEOUT
            Seconds each websocket read waits before the thread rechecks whether any job is still watched.
        """
        self._open_websocket = open_websocket
        self._clock = clock
        self._retry_interval = retry_interval
        self._receive_timeout = receive_timeout
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobState] = {}
        self._current_prompt: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._last_failure: Optional[float] = None

    def watch(self, prompt_id: str) -> None:
        """Start tracking `prompt_id`, connecting the websocket if it is not already open."""
        with self._lock:
            if prompt_id not in self._jobs:
                self._jobs[prompt_id] = _JobState(LiveProgress())
            if self._thread is not None:
                return
            if self._last_failure is not None and self._clock() - self._last_failure < self._retry_interval:
                return
            self._thread = threading.Thread(target=self._run, name='comfyui-progress', daemon=True)
            self._thread.start()

    def unwatch(self, prompt_id: str) -> None:
        """Stop tracking `prompt_id`. The thread exits after its next read once no job is watched."""
        with self._lock:
            self._jobs.pop(prompt_id, None)

    def get_progress(self, prompt_id: str) -> Optional[LiveProgress]:
        """Return a copy of the latest progress for a watched job, or None if it is not watched."""
        with self._lock:
            state = self._jobs.get(prompt_id)
            return None if state is None else LiveProgress(state.live.progress, state.live.eta_seconds,
                                                           state.live.preview)

    def handle_message(self, message: str | bytes) -> None:
        """Apply one websocket message. Messages for unwatched jobs and unknown message types are ignored."""
        if isinstance(message, bytes):
            self._handle_binary(message)
            return
        try:
            parsed = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(parsed, dict) or not isinstance(parsed.get('data'), dict):
            return
        message_type = parsed.get('type')
        data: dict[str, Any] = parsed['data']
        with self._lock:
            if message_type in ('execution_start', 'executing'):
                self._current_prompt = data.get('prompt_id', self._current_prompt)
            if message_type in _END_MESSAGES or (message_type == 'executing' and data.get('node') is None):
                self._jobs.pop(str(data.get('prompt_id')), None)
            if message_type == 'progress':
                self._apply_step(data.get('prompt_id', self._current_prompt), data)

    def _apply_step(self, prompt_id: Optional[str], data: dict[str, Any]) -> None:
        state = self._jobs.get(prompt_id) if prompt_id is not None else None
        value, maximum = data.get('value'), data.get('max')
        if (state is None or not isinstance(value, (int, float)) or not isinstance(maximum, (int, float))
                or maximum <= 0):
            return
        now = self._clock()
        node = data.get('node')
        if state.first_step is None or node != state.node or value < state.first_step[1]:
            state.node = node
            state.first_step = (now, value)
        state.live.progress = min(max(value / maximum, 0.0), 1.0)
        start_time, start_value = state.first_step
        if value > start_value:
            state.live.eta_seconds = (now - start_time) / (value - start_value) * (maximum - value)
        else:
            state.live.eta_seconds = None

    def _handle_binary(self, message: bytes) -> None:
        if len(message) < _UINT32.size * 2:
            return
        event_type = _UINT32.unpack_from(message, 0)[0]
        prompt_id: Optional[str]
        if event_type == _PREVIEW_IMAGE:
            image_bytes = message[_UINT32.size * 2:]  # The second uint32 is the image format.
            with self._lock:
                prompt_id = self._current_prompt
        elif event_type == _PREVIEW_IMAGE_WITH_METADATA:
            metadata_end = _UINT32.size * 2 + _UINT32.unpack_from(message, _UINT32.size)[0]
            try:
                metadata = json.loads(message[_UINT32.size * 2:metadata_end])
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            prompt_id = metadata.get('prompt_id') if isinstance(metadata, dict) else None
            image_bytes = message[metadata_end:]
        else:
            return
        with self._lock:
            if prompt_id not in self._jobs:
                return
        try:
            preview = image_from_bytes(image_bytes)
        except (OSError, ValueError):
            logger.debug('Ignoring an undecodable ComfyUI preview frame')
            return
        with self._lock:
            state = self._jobs.get(prompt_id) if prompt_id is not None else None
            if state is not None:
                state.live.preview = preview

    def _keep_running(self) -> bool:
        with self._lock:
            if self._jobs:
                return True
            self._thread = None
            return False

    def _run(self) -> None:
        try:
            with self._open_websocket() as connection:
                connection.settimeout(self._receive_timeout)
                while self._keep_running():
                    try:
                        message = connection.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    self.handle_message(message)
        except (SDBackendError, websocket.WebSocketException, OSError) as err:
            logger.debug('ComfyUI progress websocket closed: %s', err)
            with self._lock:
                self._last_failure = self._clock()
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
