"""Small synchronous Python SDK with ordered frames and safe frame retries."""
import base64
import hashlib
import http.client
import json
import threading
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit


class APIError(RuntimeError):
    def __init__(self, status, code, message):
        self.status, self.code = int(status), str(code)
        super().__init__("HTTP %s [%s]: %s" % (status, code, message))


class TransportError(RuntimeError):
    """A frame might already have run. Retry the identical pending PNG."""


@dataclass(frozen=True)
class FrameResult:
    data: dict

    @property
    def visible(self):
        return bool(self.data.get("mask_png") and self.data.get("ids"))

    @property
    def mask_png_bytes(self):
        value = self.data.get("mask_png")
        return None if value is None else base64.b64decode(value, validate=True)

    @property
    def mask(self):
        """HxW uint8 (0 or 255), or None. Requires OpenCV and NumPy."""
        raw = self.mask_png_bytes
        if raw is None:
            return None
        import cv2
        import numpy as np
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.isin(mask, [0, 255]).all():
            raise ValueError("Server returned an invalid binary PNG mask")
        if self.data.get("mask_sha256") != hashlib.sha256(mask.tobytes()).hexdigest():
            raise ValueError("Decoded mask SHA256 mismatch")
        return mask

    @property
    def bbox_xyxy(self):
        return self.data.get("bbox_xyxy")

    @property
    def states(self):
        return {int(identity): state for identity, state in self.data.get("states", [])}


class Client:
    def __init__(self, endpoint, backend=None, token=None, timeout=60, retries=1):
        self.endpoint = str(endpoint).rstrip("/")
        self.url = urlsplit(self.endpoint)
        if self.url.scheme not in ("http", "https") or not self.url.hostname or self.url.username or self.url.query or self.url.fragment:
            raise ValueError("Expected an HTTP(S) service endpoint")
        self.backend = backend.upper() if backend else None
        self.token, self.timeout, self.retries = token, float(timeout), int(retries)
        self.owner = "sdk-" + uuid.uuid4().hex
        self.session = None
        self.sequence = -1
        self.pending = None
        self._connection = None
        self._lock = threading.RLock()

    def _disconnect(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def _request(self, path, payload=None, retry=False):
        raw = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        attempts = 1 + self.retries if retry else 1
        with self._lock:
            for attempt in range(attempts):
                try:
                    if self._connection is None:
                        kind = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
                        self._connection = kind(self.url.hostname, self.url.port, timeout=self.timeout)
                    headers = {"Content-Type": "application/json"}
                    if self.token:
                        headers["Authorization"] = "Bearer " + self.token
                    self._connection.request("GET" if payload is None else "POST",
                        self.url.path.rstrip("/") + path, body=raw, headers=headers)
                    response = self._connection.getresponse()
                    result = json.loads(response.read())
                    if response.status != 200:
                        raise APIError(response.status, result.get("error_code", "API_ERROR"), result.get("error", response.reason))
                    return result
                except (OSError, http.client.HTTPException) as exc:
                    self._disconnect()
                    if attempt + 1 == attempts:
                        raise TransportError(str(exc)) from exc

    def health(self):
        info = self._request("/health")
        if info.get("protocol") != "flyco-text-sam-v1":
            raise ValueError("Endpoint is not a compatible A/D service")
        if self.backend and info.get("backend") != self.backend:
            raise ValueError("Expected backend %s, received %s" % (self.backend, info.get("backend")))
        return info

    def wait_ready(self, seconds=120):
        deadline = time.monotonic() + float(seconds)
        while True:
            try:
                info = self.health()
            except TransportError:
                if time.monotonic() >= deadline:raise
                time.sleep(.25)
                continue
            if info["status"] == "ready":
                return info
            if info["status"] == "error":
                raise RuntimeError(info.get("error", "Model loading failed"))
            if time.monotonic() >= deadline:
                raise TimeoutError("Model did not become ready")
            time.sleep(.25)

    def start(self, text):
        """Create a target session; when already open, explicitly switch the target."""
        with self._lock:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Enter a target description")
            info = self.health()
            if info["status"] != "ready":
                raise RuntimeError("Model is not ready")
            result = self._request("/session", {"text": text, "owner": self.owner,
                                               "replace_session": self.session if info.get('busy') else None})
            self.session, self.sequence, self.pending = result["session"], -1, None
            return result

    def process_png(self, image_png, timestamp_ns=None):
        with self._lock:
            if self.session is None:
                raise RuntimeError("Call start(text) before sending frames")
            raw = bytes(image_png)
            payload = {"session": self.session, "sequence": self.sequence + 1,
                       "image_png": base64.b64encode(raw).decode("ascii")}
            if timestamp_ns is not None:
                if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
                    raise ValueError("timestamp_ns must be a nonnegative integer")
                payload["timestamp_ns"] = timestamp_ns
            if self.pending is not None and payload != self.pending:
                raise RuntimeError("Previous response is uncertain; call retry_pending() with the identical frame, or reset the session")
            self.pending = payload
            return self.retry_pending()

    def retry_pending(self):
        with self._lock:
            if self.pending is None:
                raise RuntimeError("There is no pending frame")
            payload = self.pending
            try:
                result = self._request("/frame", payload, retry=True)
            except APIError as exc:
                if exc.status in (400, 401, 404, 409, 413, 422):
                    self.pending = None
                if exc.code=='UNKNOWN_SESSION':
                    self.session,self.sequence=None,-1
                raise
            if result.get("session") != self.session or result.get("sequence") != payload["sequence"]:
                raise RuntimeError("Stale or out-of-order inference response")
            if result.get("timestamp_ns") != payload.get("timestamp_ns"):
                raise RuntimeError("Response timestamp differs from the submitted frame")
            self.sequence, self.pending = payload["sequence"], None
            return FrameResult(result)

    def process_rgb(self, image, timestamp_ns=None):
        import cv2
        import numpy as np
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected HxWx3 uint8 RGB")
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("PNG encoding failed")
        result = self.process_png(encoded.tobytes(), timestamp_ns)
        if result.data.get("input_rgb_sha256") != hashlib.sha256(image.tobytes()).hexdigest():
            self.reset()
            raise ValueError("Server received different RGB pixels")
        return result

    def process_bgr(self, image, timestamp_ns=None):
        import cv2
        return self.process_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), timestamp_ns)

    def hint(self, reason="external_geometry_hint"):
        if self.session is None:
            raise RuntimeError("No active session")
        if self.pending is not None:
            raise RuntimeError('Resolve the pending frame before submitting a hint')
        return self._request("/hint", {"session": self.session, "reason": reason})

    def reset(self):
        with self._lock:
            if self.session is not None:
                try:
                    self._request("/reset", {"session": self.session})
                except APIError as exc:
                    if exc.code!='UNKNOWN_SESSION':raise
            self.session, self.sequence, self.pending = None, -1, None

    def close(self):
        try:
            self.reset()
        finally:
            self._disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
