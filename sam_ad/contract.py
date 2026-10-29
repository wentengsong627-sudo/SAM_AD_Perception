"""Validation and response additions around the unchanged A/D model lifecycle."""
import base64
import hashlib
import json
import struct
import time


class RequestError(ValueError):
    def __init__(self, code, message, status=400):
        self.code, self.status = code, status
        super().__init__(message)


def png_size(raw):
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise RequestError("INVALID_PNG", "image_png must encode a PNG image")
    w, h = struct.unpack(">II", raw[16:24])
    if not 0 < w <= 4096 or not 0 < h <= 4096 or w*h > 4096*2160:
        raise RequestError("IMAGE_TOO_LARGE", "Maximum 8,847,360 pixels; each edge <= 4096", 413)
    return w, h


class Contract:
    def __init__(self, idle_seconds=180):
        self.idle_seconds = float(idle_seconds)
        self.touched = time.monotonic()
        self.shape = None
        self.request_key = None
        self.response = None

    def expire(self, host):
        if host.session and self.idle_seconds > 0 and time.monotonic()-self.touched > self.idle_seconds:
            host.dispatch("/reset", {"session": host.session})
            self.shape = self.request_key = self.response = None
            return True
        return False

    def dispatch(self, host, path, data):
        expired_session, expired_owner = host.session, host.owner
        expired = self.expire(host)
        if not isinstance(data, dict):
            raise RequestError("INVALID_JSON", "Expected a JSON object")
        if path == "/session":
            # The lease can expire between the client's health check and this request.
            if expired and data.get('replace_session') == expired_session and data.get('owner') == expired_owner:
                data = dict(data, replace_session=None)
            if not isinstance(data.get("text"), str) or not 0 < len(data["text"].strip()) <= 200:
                raise RequestError("INVALID_TEXT", "text must be a nonempty string of at most 200 characters", 422)
            if not isinstance(data.get("owner"), str) or not 1 <= len(data["owner"]) <= 128:
                raise RequestError("INVALID_OWNER", "owner must be a nonempty string of at most 128 characters", 422)
            # Apply exactly the same parser as the real application before resetting an old session.
            command = host.parse_command(data["text"])
            if command.action != "track" or not command.query.strip(" ."):
                raise RequestError("INVALID_TEXT", "Use /reset to stop; /session requires a target", 422)
            if command.quantity != "one":
                raise RequestError("SINGLE_TARGET_ONLY", "This interface tracks one selected instance per session", 422)
            if host.backend == "D":
                tokenizer = host.engine.model.backbone.language_backbone.tokenizer
                if len(tokenizer.encode(command.query.rstrip(".")))+2 > 16:
                    raise RequestError("TEXT_TOO_LONG", "D supports at most 14 MobileCLIP content tokens", 422)
            result = host.dispatch(path, data)
            self.shape = self.request_key = self.response = None
            self.touched = time.monotonic()
            return dict(result, query=command.query, max_objects=1, sequence_start=0,
                        idle_timeout_seconds=self.idle_seconds)
        if path != "/frame":
            if not host.session or data.get('session') != host.session:
                raise RequestError('UNKNOWN_SESSION','Session is expired or unknown',409)
            result = host.dispatch(path, data)
            self.touched = time.monotonic()
            if path == "/reset":
                self.shape = self.request_key = self.response = None
            elif path == "/hint" and self.response is not None:
                self.response = dict(self.response, mask_png=None, mask_sha256=None,
                                     bbox_xyxy=None, visible=False, visible_pixels=0, ids=[], phase='lost',
                                     states=[[t['id'],t['state']] for t in host.pipe.tracks.values()])
            return result
        if data.get("session") != host.session or host.session is None:
            raise RequestError("UNKNOWN_SESSION", "Session is expired or unknown", 409)
        sequence = data.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise RequestError("INVALID_SEQUENCE", "sequence must be a nonnegative integer")
        timestamp = data.get("timestamp_ns")
        if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0):
            raise RequestError("INVALID_TIMESTAMP", "timestamp_ns must be a nonnegative integer")
        try:
            raw = base64.b64decode(data["image_png"], validate=True)
        except Exception as exc:
            raise RequestError("INVALID_PNG", "image_png must be valid base64 PNG") from exc
        width, height = png_size(raw)
        key = hashlib.sha256(raw + json.dumps([timestamp],separators=(",", ":")).encode()).hexdigest()
        if sequence == host.sequence:
            if key != self.request_key:
                raise RequestError("FRAME_CONFLICT", "A repeated sequence must contain identical PNG and timestamp", 409)
            self.touched = time.monotonic()
            return dict(self.response)
        if sequence != host.sequence+1:
            raise RequestError("OUT_OF_ORDER", "Expected sequence %d" % (host.sequence+1), 409)
        if self.shape and self.shape != (width, height):
            raise RequestError("IMAGE_SIZE_CHANGED", "Open a new session before changing image dimensions", 409)
        self.touched = time.monotonic()
        result = dict(host.dispatch(path, data))
        box, pixels = None, 0
        if result.get("mask_png"):
            import cv2
            import numpy as np
            mask = cv2.imdecode(np.frombuffer(base64.b64decode(result["mask_png"]),np.uint8),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height,width) or not np.isin(mask,[0,255]).all():
                raise RuntimeError("Model mask contract is broken")
            pixels = int(np.count_nonzero(mask))
            if pixels:
                x,y,w,h = cv2.boundingRect(mask)
                box = [int(x),int(y),int(x+w),int(y+h)]
        result.update(api_version="1.0", timestamp_ns=timestamp,
                      image={"width":width,"height":height,"color_order":"RGB"},
                      visible=bool(pixels),visible_pixels=pixels,bbox_xyxy=box,
                      bbox_convention="xyxy_pixels_max_exclusive",prediction_only=False)
        self.shape, self.request_key, self.response = (width,height), key, result
        self.touched = time.monotonic()
        return result
