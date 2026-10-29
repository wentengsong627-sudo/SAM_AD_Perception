"""Single-target text provider; usable by ROS Noetic without model dependencies.

``initialized`` means the text session is armed. A reliable visible target exists
only when track() returns a nonempty uint8 mask. Geometry hints preserve identity.
"""
import base64
import hashlib
import http.client
import json
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

PROTOCOL = "flyco-text-sam-v1"
REVISION = "20261026-r3"


@dataclass(frozen=True)
class HTTPProviderStats:
    initialized: bool
    initialization_count: int
    reinitialization_count: int
    reset_count: int
    last_init_reason: str
    has_last_prompt: bool
    backend: str
    frame_count: int
    depth_hint_count: int
    has_reliable_mask: bool


class HTTPTextSAMProvider:
    def __init__(self, endpoint, backend, timeout=30):
        self.endpoint = str(endpoint).rstrip("/")
        self.backend = str(backend).upper()
        if self.backend not in ("A", "D"):
            raise ValueError("backend must be A or D")
        self.timeout = float(timeout)
        url = urlsplit(self.endpoint)
        if url.scheme not in ("http", "https") or not url.hostname or url.username:
            raise ValueError("Expected an HTTP(S) model service endpoint")
        self._url = url
        self._connection = None
        self.owner = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._session = None
        self._sequence = -1
        self._last_prompt = None
        self._last_frame = self._last_mask = None
        self._frame_token = None
        self._initializations = self._reinitializations = self._resets = 0
        self._frames = self._depth_hints = 0
        self._last_reason = ""
        self.last_metadata = {}
        self.last_reinit_deferred = False
        health = self._request("/health")
        if (health.get("protocol") != PROTOCOL or
                health.get("service_revision") != REVISION or
                health.get("backend") != self.backend):
            self._disconnect()
            raise ValueError("Wrong inference service backend/version: " + str(health))
        if health.get("status") != "ready":
            self._disconnect()
            raise RuntimeError("Inference service is not ready: " + str(health))

    def _disconnect(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        attempts = 2 if path == "/frame" else 1
        with self._lock:
            for attempt in range(attempts):
                try:
                    if self._connection is None:
                        connection = (http.client.HTTPSConnection if self._url.scheme == "https"
                                      else http.client.HTTPConnection)
                        self._connection = connection(self._url.hostname, self._url.port, timeout=self.timeout)
                    self._connection.request("GET" if data is None else "POST",
                                             self._url.path.rstrip("/") + path, body=data,
                                             headers={"Content-Type": "application/json"})
                    response = self._connection.getresponse()
                    content = response.read()
                    result = json.loads(content)
                    if response.status != 200:
                        raise RuntimeError("Inference HTTP %s: %s" % (
                            response.status, result.get("error", response.reason)))
                    return result
                except (OSError, http.client.HTTPException):
                    self._disconnect()
                    if attempt + 1 == attempts:
                        raise

    @property
    def initialized(self):
        return self._session is not None

    @property
    def last_prompt(self):
        return self._last_prompt

    @property
    def stats(self):
        return HTTPProviderStats(
            self.initialized, self._initializations, self._reinitializations,
            self._resets, self._last_reason, self._last_prompt is not None,
            self.backend, self._frames, self._depth_hints, self._last_mask is not None)

    def begin_frame(self, token):
        """Allows callers that reuse an RGB buffer to identify a new source frame."""
        with self._lock:
            if token != self._frame_token:
                self._last_frame = None
                self._frame_token = token

    def can_initialize_from_prompt(self, prompt):
        return bool(prompt and str(prompt.get("text", "")).strip()
                    and np.asarray(prompt.get("coords", [])).size == 0)

    def initialize(self, frame, prompt):
        with self._lock:
            if not self.can_initialize_from_prompt(prompt):
                raise ValueError("A/D accepts one text target. Use native for click prompts.")
            result = self._request("/session", {
                "text": str(prompt["text"]).strip(), "owner": self.owner,
                "replace_session": self._session})
            self._session, self._sequence = result["session"], -1
            self._last_frame = self._last_mask = None
            self._last_prompt = dict(prompt)
            self.last_reinit_deferred = False
            self._initializations += 1
            self._last_reason = "initialize"
            return self.track(frame)

    def reinitialize_from_prompt(self, frame, prompt, reason="reinitialize"):
        with self._lock:
            if str(reason).startswith("depth_"):
                if not self.initialized:
                    return None
                text = str(prompt.get("text", "")).strip()
                if text and text != str(self._last_prompt["text"]).strip():
                    raise ValueError("Geometry cannot change the target text")
                self.last_metadata = self._request("/hint", {
                    "session": self._session, "reason": str(reason)})
                self.last_reinit_deferred = True
                self._depth_hints += 1
                self._last_reason = "depth_hint_deferred"
                self._last_frame, self._last_mask = frame, None
                return None
            mask = self.initialize(frame, prompt)
            self._reinitializations += 1
            self._last_reason = str(reason)
            return mask

    def track(self, frame, depth_mask=None):
        with self._lock:
            if not self.initialized:
                return None
            # FlyCo initializes then tracks the SAME RGB object in one callback.
            if frame is self._last_frame:
                return None if self._last_mask is None else self._last_mask.copy()
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Expected HxWx3 uint8 RGB")
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("RGB PNG encoding failed")
            sequence = self._sequence + 1
            self._last_mask = None
            result = self._request("/frame", {
                "session": self._session, "sequence": sequence,
                "image_png": base64.b64encode(encoded).decode()})
            if result.get("session") != self._session or result.get("sequence") != sequence:
                raise RuntimeError("Stale/out-of-order inference response")
            if result.get("input_rgb_sha256") != hashlib.sha256(frame.tobytes()).hexdigest():
                raise RuntimeError("Inference service received different RGB pixels")
            mask = None
            if result.get("mask_png") is not None:
                mask = cv2.imdecode(np.frombuffer(
                    base64.b64decode(result["mask_png"]), np.uint8), cv2.IMREAD_GRAYSCALE)
                if (mask is None or mask.shape != frame.shape[:2] or
                        not np.isin(mask, [0, 255]).all()):
                    raise RuntimeError("Invalid binary mask from inference service")
                if not mask.any():
                    mask = None
            actual_mask_hash = None if mask is None else hashlib.sha256(mask.tobytes()).hexdigest()
            if result.get("mask_sha256") != actual_mask_hash:
                raise RuntimeError("Decoded mask differs from the service inference result")
            self._sequence = sequence
            self._frames += 1
            self._last_frame, self._last_mask = frame, mask
            self.last_metadata = result
            return None if mask is None else mask.copy()

    def reset(self):
        with self._lock:
            try:
                if self._session is not None:
                    self._request("/reset", {"session": self._session})
            finally:
                self._disconnect()
                self._session = None
                self._sequence = -1
                self._last_frame = self._last_mask = self._last_prompt = None
                self.last_reinit_deferred = False
                self._resets += 1
                self._last_reason = "reset"
