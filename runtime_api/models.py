"""Construct original local A/D models; keep all GPU work on one owner thread."""
import hashlib
import inspect
import os
import sys
import textwrap
import time
from contextlib import nullcontext
from pathlib import Path


def build_models(backend, workspace, app_root):
    root, application = Path(workspace).resolve(), Path(app_root).resolve()
    if not (application / "text_sam/pipeline.py").is_file():
        raise ValueError("app_root must contain the guarded text_sam and abc_sam application")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", SAM_DATA_ROOT=str(root),
        SAM_DEVICE="cuda", SAM_CPU_THREADS="4", SAM_RECOVERY_GATE="0",
        SAM_RECOVERY_PARTIAL_RETRY="0", TOKENIZERS_PARALLELISM="false")
    sys.path[:0] = [str(application), str(root / "sam2-official")]
    import torch
    from text_sam import vision
    vision.ROOT = root
    vision.configure_runtime()
    from text_sam.pipeline import TrackingPipeline
    from text_sam.commands import parse_command
    if not torch.cuda.is_available():
        raise RuntimeError("A/D inference requires CUDA; no silent CPU fallback")
    ground = None
    if backend == "A":
        ground = vision.Grounder(device="cuda")
        engine = vision.StreamingSAM2("tiny", 1024, device="cuda", decoder_batch_size=1)
        engine.autocast = lambda: nullcontext()
        ground.model.float()

        def make_pipe():
            return TrackingPipeline(ground, engine, .3, 30, 1, 90, 1, True)

        size = 1024
    elif backend == "D":
        import abc_sam.light_models as light
        from abc_sam.image_tracking import GuardedImagePipeline
        # Override only our app's precision context, leaving upstream code/weights intact.
        source = textwrap.dedent(inspect.getsource(light.LightImageEngine.segment))
        before = 'with torch.autocast("cuda", dtype=torch.bfloat16):'
        if before not in source:
            raise RuntimeError("Unsupported image adapter revision")
        scope = dict(light.__dict__)
        exec(compile(source.replace(before, 'with torch.autocast("cuda", enabled=False):'),
                     "<flyco_ad_FP32_adapter>", "exec"), scope)
        light.LightImageEngine.segment = scope["segment"]
        engine = light.LightImageEngine("D")

        def make_pipe():
            return GuardedImagePipeline(None, engine, .3, 30, 1, 90, 1, True)

        size = engine.actual_size
    else:
        raise ValueError("backend must be A or D")
    engine.model.float()
    # Model builders can enable TF32 during import. Set AND assert after loading.
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert all(p.dtype == torch.float32 for p in engine.model.parameters())
    # Run cold kernel setup before advertising readiness. Synthetic warmup never
    # establishes application identities, and raw tracker memory is cleared.
    import cv2
    import numpy as np
    tick = time.perf_counter()
    warm_rgb = np.full((480, 640, 3), 127, dtype=np.uint8)
    with torch.inference_mode():
        if backend == "A":
            ground.locate_guarded(warm_rgb, parse_command("dog"), .3)
            engine.reset()
            for index in range(8):
                engine.step_many(warm_rgb, index, 1_000_000_000,
                                 {1: [160, 120, 480, 360]} if index == 0 else None)
            engine.reset()
            assert engine.history_frames == 0 and engine.last_frame is None
        else:
            for _ in range(8):
                engine.segment(warm_rgb, "dog", .3)
            engine.calls = 0
        torch.cuda.synchronize()
    metadata = {
        "backend": backend, "device": torch.cuda.get_device_name(), "precision": "FP32",
        "matmul_tf32": False, "cudnn_tf32": False, "autocast": False,
        "model_input_size": size, "stride": 1, "max_objects": 1,
        "temporal_memory_storage": "BF16 cached SAM2 features" if backend == "A" else "no neural video memory",
        "warmup_complete": True, "warmup_frames": 8,
        "warmup_ms": (time.perf_counter() - tick) * 1000,
        "warmup_identity_memory_cleared": True,
        "workspace": str(root), "app_root": str(application),
        "source_hashes": {str(p.relative_to(application)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [application / "text_sam/pipeline.py", application / "text_sam/vision.py",
                                    application / "abc_sam/image_tracking.py"]},
    }
    return make_pipe, parse_command, engine, ground, metadata
