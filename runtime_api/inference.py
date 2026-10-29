"""Stateful model host and bounded executor, shared by HTTP and regression tests."""
import base64
import hashlib
import queue
import threading
import time
import uuid
from concurrent.futures import Future

import cv2
import numpy as np


class ProtocolError(ValueError):
    pass


class InferenceHost:
    def __init__(self, backend, make_pipe, parse_command, engine, metadata):
        self.backend, self.make_pipe, self.parse_command = backend, make_pipe, parse_command
        self.engine, self.metadata = engine, metadata
        self.pipe = self.command = self.session = self.owner = None
        self.sequence = -1
        self.last_digest = self.last_response = None
        self.process_calls = 0

    def _check(self, payload):
        if self.session is None or payload.get("session") != self.session:
            raise ProtocolError("Unknown or expired session")

    def dispatch(self, path, payload):
        import torch
        with torch.inference_mode():
            if path == "/session":
                command = self.parse_command(str(payload["text"]))
                if command.quantity != "one":
                    raise ProtocolError("FlyCo currently exposes one target cloud; 'all' tasks are unsupported")
                owner, previous = str(payload["owner"]), payload.get("replace_session")
                if self.session is not None and (self.owner != owner or previous != self.session):
                    raise ProtocolError("Inference is busy with another session; stop that task first")
                if previous is not None and previous != self.session:
                    raise ProtocolError("Expired replacement session")
                next_id = self.pipe.next_id if previous is not None and self.pipe is not None else 1
                self.pipe = self.make_pipe()
                self.pipe.next_id = next_id
                self.command, self.owner = command, owner
                self.session, self.sequence = uuid.uuid4().hex, -1
                self.last_digest = self.last_response = None
                return {"session": self.session, "backend": self.backend}
            self._check(payload)
            if path == "/reset":
                self.pipe.tracker.reset()
                self.pipe = self.command = self.session = self.owner = None
                self.last_digest = self.last_response = None
                return {"reset": True}
            if path == "/hint":
                references = [i for i, t in self.pipe.tracks.items() if t.get("verified")]
                for track in self.pipe.tracks.values():
                    if track.get("verified") and track["state"] == "TRACKING":
                        self.pipe.state(track, "SUSPECT", max(0, self.sequence), "external_depth_hint")
                self.pipe.last_detection_frame = None
                if self.last_response is not None:
                    self.last_response = dict(self.last_response, mask_png=None)
                return {"deferred": True, "reference_ids": references, "backend": self.backend}
            if path != "/frame":
                raise ProtocolError("Unknown inference endpoint")
            raw = base64.b64decode(payload["image_png"], validate=True)
            digest, sequence = hashlib.sha256(raw).hexdigest(), int(payload["sequence"])
            if sequence == self.sequence:
                if digest != self.last_digest:
                    raise ProtocolError("Repeated frame number contains different RGB data")
                return self.last_response
            if sequence != self.sequence + 1:
                raise ProtocolError("Expected the next consecutive frame")
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[0] * bgr.shape[1] > 4096 * 2160:
                raise ProtocolError("Invalid or oversized RGB image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            tick = time.perf_counter()
            snap = self.pipe.process(rgb, sequence, 1_000_000_000,
                                     self.command if sequence == 0 else None)
            self.process_calls += 1
            assert len(snap["masks"]) <= 1
            mask = next(iter(snap["masks"].values()), None)
            encoded = mask_digest = None
            if mask is not None:
                binary = mask.astype(np.uint8) * 255
                mask_digest = hashlib.sha256(binary.tobytes()).hexdigest()
                ok, buf = cv2.imencode(".png", binary)
                if not ok:
                    raise RuntimeError("Mask PNG encoding failed")
                encoded = base64.b64encode(buf).decode()
            result = {
                "session": self.session, "sequence": sequence, "mask_png": encoded,
                "input_rgb_sha256": rgb_digest, "mask_sha256": mask_digest,
                "ids": sorted(int(i) for i in snap["masks"]), "phase": snap["phase"],
                "states": [[t["id"], t["state"]] for t in snap["objects"]],
                "memory_updates": self.pipe.memory_updates,
                "identity_commits": int(getattr(self.engine, "identity_commits", self.pipe.memory_updates)),
                "detection_calls": self.pipe.detection_count,
                "sam_encoder_calls": self.pipe.sam_encoder_calls,
                "image_calls": int(getattr(self.engine, "calls", 0)),
                "process_calls": self.process_calls,
                "server_model_ms": (time.perf_counter() - tick) * 1000,
                "backend": self.backend,
            }
            self.sequence, self.last_digest, self.last_response = sequence, digest, result
            return result


class ModelExecutor:
    """Loads and runs one model on one persistent thread; HTTP threads do I/O only."""
    def __init__(self, backend, workspace, app_root):
        self.tasks = queue.Queue(maxsize=4)
        self.ready = threading.Event()
        self.error = None
        self.host = None
        self.thread = threading.Thread(target=self._run, args=(backend, workspace, app_root),
                                       name="flyco-model-owner", daemon=True)
        self.thread.start()

    def _run(self, backend, workspace, app_root):
        try:
            from .models import build_models
            factory, parse, engine, ground, metadata = build_models(backend, workspace, app_root)
            self.host = InferenceHost(backend, factory, parse, engine, metadata)
            self.host.ground = ground
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()
        if self.error is not None:
            return
        while True:
            job = self.tasks.get()
            if job is None:
                return
            function, future = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                import torch
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                with torch.inference_mode():
                    future.set_result(function(self.host))
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, function):
        if not self.ready.is_set():
            raise RuntimeError("Model is loading")
        if self.error is not None:
            raise RuntimeError("Model loading failed: " + str(self.error)) from self.error
        future = Future()
        try:
            self.tasks.put_nowait((function, future))
        except queue.Full as exc:
            raise RuntimeError("Inference queue is busy; no unbounded frame buffering") from exc
        return future

    def call(self, path, payload, timeout=60):
        return self.submit(lambda host: host.dispatch(path, payload)).result(timeout=timeout)

    def close(self):
        if self.thread.is_alive():
            self.tasks.put(None, timeout=5)
            self.thread.join(timeout=5)
