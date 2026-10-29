"""Text grounding plus bounded-memory, one-frame-at-a-time SAM 2 inference.

Uses SAM2Base.track_step from upstream commit 2b90b9f. No upstream files are
modified. Frame history keeps the conditioning frame and the last 32 results;
this covers the default six recent spatial memories and 16 object pointers.
Objects share one image encoder; compatible decoder histories run in batches.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import time
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import runtime
from .commands import Command
from .targets import select_targets

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SAM_DATA_ROOT", str(ROOT))).resolve()


def configure_runtime():
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get("SAM_CPU_THREADS", "4")))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


class Grounder:
    def __init__(self, device=None, fast_shapes=True):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        folder = DATA_ROOT / "models" / "grounding-dino-tiny"
        device = device or runtime.require_device()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        load_options={}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            folder, local_files_only=True, use_safetensors=True,
            disable_custom_kernels=True,**load_options,
        ).to(device).eval()
        self.recovery_gate_enabled=os.environ.get('SAM_RECOVERY_GATE','0')=='1'
        self.recovery_partial_retry=os.environ.get('SAM_RECOVERY_PARTIAL_RETRY','0')=='1'
        self.recovery_gate_max_gap=int(os.environ.get('SAM_RECOVERY_GATE_MAX_GAP','60'))
        detector_edge=os.environ.get('SAM_DINO_SHORT_EDGE')
        if detector_edge:
            if detector_edge not in {'512','800'}:raise ValueError('Board detector supports tested 512 or original 800 input')
            if detector_edge=='512':self.processor.image_processor.size={'shortest_edge':512,'longest_edge':854}
        self.shape_cache = {}
        if fast_shapes:
            from .fast_grounding import enable_fast_shapes
            self.shape_cache, _ = enable_fast_shapes(self.model)
        pass  # Board-only path omitted from this CUDA release.

    @torch.inference_mode()
    def locate(self, rgb, command: Command, threshold=0.30):
        started = time.perf_counter()
        self.shape_cache.clear()
        inputs = self.processor(images=rgb, text=command.query, return_tensors="pt").to(self.device)
        # Keep the detector in float32; SAM 2 uses a separate autocast context.
        outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=threshold, text_threshold=0.25,
            target_sizes=[rgb.shape[:2]],
        )[0]
        boxes = result["boxes"].float()
        scores = result["scores"].float()
        h, w = rgb.shape[:2]
        valid = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
        boxes, scores = boxes[valid], scores[valid]
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
        boxes, scores = boxes[valid], scores[valid]
        from torchvision.ops import nms
        keep = nms(boxes, scores, 0.5)
        boxes, scores = boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()
        selection = select_targets(boxes, scores, command, w, h, threshold)
        selection["grounding_ms"] = (time.perf_counter() - started) * 1000
        return selection

    @torch.inference_mode()
    def locate_guarded(self,rgb,command: Command,threshold=.30):
        """Category contrast is supporting evidence, followed by masked identity checks."""
        started=time.perf_counter();self.shape_cache.clear()
        categories={'ball':['cup','person'],'dog':['person','cat'],'car':['person','cup'],
                    'cup':['ball','person'],'person':['dog','cup'],'hippopotamus':['person','cup'],
                    'zebra':['person','cup']}
        target=command.query.strip().rstrip('.')
        noun=next((n for n in categories if re.search(r'\b'+n+r'\b',target)),None)
        rivals=categories.get(noun,['person','car','cup'])
        rivals=[r for r in rivals if not re.search(r'\b'+r+r'\b',target)]
        query=target+'. '+'. '.join(rivals)+'.'
        inputs=self.processor(images=rgb,text=query,return_tensors='pt').to(self.device)
        output=self.model(**inputs)
        probabilities=output.logits[0].sigmoid()
        prefix=self.processor.tokenizer(target+'.',add_special_tokens=False)['input_ids']
        target_indices=[n+1 for n,t in enumerate(prefix) if t not in self.processor.tokenizer.all_special_ids and self.processor.tokenizer.convert_ids_to_tokens(t)!='.']
        start=1+len(prefix);rival_indices=[]
        for rival in rivals:
            ids=self.processor.tokenizer(rival+'.',add_special_tokens=False)['input_ids']
            rival_indices.extend(start+n for n,t in enumerate(ids) if self.processor.tokenizer.convert_ids_to_tokens(t)!='.')
            start+=len(ids)
        scores=probabilities[:,target_indices].max(-1).values
        contrary=probabilities[:,rival_indices].max(-1).values if rival_indices else torch.zeros_like(scores)
        original_logits=output.logits
        output.logits=original_logits.clone()
        token_mask=torch.ones(output.logits.shape[-1],dtype=torch.bool,device=self.device);token_mask[target_indices]=False
        output.logits[:,:,token_mask]=-100
        result=self.processor.post_process_grounded_object_detection(output,inputs.input_ids,box_threshold=threshold,text_threshold=.25,target_sizes=[rgb.shape[:2]])[0]
        raw_indices=torch.where(scores>threshold)[0]
        boxes=result['boxes'].float();box_scores=result['scores'].float()
        h,w=rgb.shape[:2];boxes[:,0::2].clamp_(0,w);boxes[:,1::2].clamp_(0,h)
        valid=torch.isfinite(boxes).all(-1)&((boxes[:,2]-boxes[:,0])>=2)&((boxes[:,3]-boxes[:,1])>=2)
        boxes,box_scores,raw_indices=boxes[valid],box_scores[valid],raw_indices[valid]
        from torchvision.ops import nms
        keep=nms(boxes,box_scores,.5)
        evidence={}
        for box,index in zip(boxes[keep].cpu().tolist(),raw_indices[keep].cpu().tolist()):
            a,b=float(scores[index]),float(contrary[index])
            evidence[tuple(box)]={'target_score':a,'rival_score':b,'margin':a-b,'rivals':rivals}
        selection=select_targets(boxes[keep].cpu().numpy(),box_scores[keep].cpu().numpy(),command,w,h,threshold)
        for c in selection['candidates']:
            e=evidence[tuple(c['box'])];c['category_evidence']=e
            c['category_ok']=e['target_score']>e['rival_score']+.04
        top=torch.maximum(scores,contrary).argsort(descending=True)[:8].cpu().tolist()
        selection['category_audit']=[{'target_score':float(scores[i]),'rival_score':float(contrary[i])} for i in top]
        selection['grounding_ms']=(time.perf_counter()-started)*1000
        selection['detection_text']=query
        return selection


class StreamingSAM2:
    def __init__(self, model_name="tiny", image_size=768, device=None, history_limit=32, decoder_batch_size=4):
        from sam2.build_sam import build_sam2_video_predictor
        device = device or runtime.require_device()
        if model_name not in {"tiny", "small"} or image_size not in {256,512,768,1024}:
            raise ValueError("Unsupported model or input size")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用 CUDA。请先运行 GPU 检查，不能用 CPU 冒充 GPU 性能测试。")
        letter = "t" if model_name == "tiny" else "s"
        builder=build_sam2_video_predictor
        pass  # Board-only path omitted from this CUDA release.
        self.model = builder(
            f"configs/sam2.1/sam2.1_hiera_{letter}.yaml",
            str(DATA_ROOT / "models" / f"sam2.1_hiera_{model_name}.pt"),
            device=device, apply_postprocessing=False,
            hydra_overrides_extra=[f"++model.image_size={image_size}", "++model.max_cond_frames_in_attn=1"],
        )
        self.device, self.image_size = device, image_size
        # The laptop's 768+ BF16 batch path showed rounding drift without a
        # measurable speed benefit; keep that path equivalent to single decoding.
        self.decoder_batch_size = max(1, min(8, decoder_batch_size)) if image_size == 512 else 1
        self.history_limit = max(history_limit, self.model.max_obj_ptrs_in_encoder, self.model.num_maskmem)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.npu_encoder = None
        pass  # Board-only path omitted from this CUDA release.
        self.npu_memory = None
        pass  # Board-only path omitted from this CUDA release.
        self.ort_memory = None
        pass  # Board-only path omitted from this CUDA release.
        self.npu_memory_encoder = None
        pass  # Board-only path omitted from this CUDA release.
        self.reset()
        pass  # Board-only path omitted from this CUDA release.

    def reset(self):
        self.histories = {}
        self.pending = {}
        self.last_committed_ids = []
        self.last_rejected_ids = []
        self.batch_eligible = {}
        self.history = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self.last_frame = None

    def drop_objects(self, object_ids):
        for object_id in object_ids:
            self.pending.pop(object_id, None)
            self.histories.pop(object_id, None)
            self.batch_eligible.pop(object_id, None)
        self.history = next(iter(self.histories.values()), {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}})

    @property
    def history_frames(self):
        return sum(len(h["cond_frame_outputs"]) + len(h["non_cond_frame_outputs"]) for h in self.histories.values())

    def _merge_histories(self, object_ids, histories=None):
        histories = self.histories if histories is None else histories
        if len(object_ids) == 1:
            return histories[object_ids[0]]
        merged = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        for bucket, entries in histories[object_ids[0]].items():
            for index in entries:
                rows = [histories[i][bucket][index] for i in object_ids]
                packed = {}
                for key, value in rows[0].items():
                    if isinstance(value, list):
                        packed[key] = [torch.cat([r[key][n] for r in rows], dim=0) for n in range(len(value))]
                    else:
                        packed[key] = torch.cat([r[key] for r in rows], dim=0) if value is not None else None
                merged[bucket][index] = packed
        return merged

    def preprocess(self, rgb):
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        return (image / 255.0 - self.mean) / self.std

    @contextmanager
    def autocast(self):
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        attention = nullcontext()
        if self.device == "cuda" and os.environ.get("SAM2_ATTENTION_BACKEND") == "efficient":
            from torch.nn.attention import sdpa_kernel, SDPBackend
            attention = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)
        with precision, attention:
            yield

    @torch.inference_mode()
    def step(self, rgb, frame_index, total_frames, box=None):
        """Legacy single-object interface used by the upstream equivalence test."""
        if box is None and self.last_frame is None:
            raise RuntimeError("先用文字生成目标框，再开始跟踪。")
        if box is not None:
            self.reset()
        outputs = self.step_many(rgb, frame_index, total_frames, {1: box} if box is not None else {})
        return outputs[1]

    @torch.inference_mode()
    def step_many(self, rgb, frame_index, total_frames, boxes=None):
        """Compatible immediate-commit API; guarded applications use predict_many."""
        results = self.predict_many(rgb, frame_index, total_frames, boxes)
        self.commit_many(results)
        return results

    def commit_many(self, accepted_ids):
        accepted = set(accepted_ids)
        self.last_committed_ids = sorted(accepted & self.pending.keys())
        self.last_rejected_ids = sorted(self.pending.keys() - accepted)
        for object_id in self.last_committed_ids:
            index, initial, compact, eligible = self.pending[object_id]
            if initial:
                self.histories[object_id] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            history = self.histories[object_id]
            history["cond_frame_outputs" if initial else "non_cond_frame_outputs"][index] = compact
            for old in list(history["non_cond_frame_outputs"]):
                if old < index-self.history_limit+1:
                    del history["non_cond_frame_outputs"][old]
            self.batch_eligible[object_id] = eligible
        self.pending.clear()
        self.history = next(iter(self.histories.values()), {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}})

    @torch.inference_mode()
    def predict_many(self, rgb, frame_index, total_frames, boxes=None):
        """Encode a frame once; keep independent memory and masks per object ID."""
        boxes = boxes or {}
        self.pending.clear()
        self.last_committed_ids = []
        self.last_rejected_ids = []
        if self.last_frame is not None and frame_index != self.last_frame + 1:
            raise RuntimeError("跟踪必须按连续帧顺序运行。")
        h, w = rgb.shape[:2]
        histories = dict(self.histories)
        eligibility = dict(self.batch_eligible)
        for object_id in boxes:
            histories[object_id] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            x1, y1, x2, y2 = boxes[object_id]
            eligibility[object_id] = min((x2 - x1) * self.image_size / w, (y2 - y1) * self.image_size / h) >= 16
        results = {}
        if not histories:
            self.last_frame = frame_index
            return results
        with self.autocast():
            image = self.preprocess(rgb)
            prefetch=getattr(self,'image_prefetch',None)
            backbone = prefetch.encode(self.encoder_frame_key,image) if prefetch else self.npu_encoder(image) if self.npu_encoder else self.model.forward_image(image)
            _, features, positions, sizes = self.model._prepare_backbone_features(backbone)
            groups = {}
            for object_id, history in histories.items():
                signature = (object_id in boxes, tuple(history["cond_frame_outputs"]), tuple(history["non_cond_frame_outputs"]))
                # Very small masks are sensitive to BF16 batch-kernel rounding.
                # Preserve independent decoding for them and batch larger targets.
                if object_id in boxes or self.model.non_overlap_masks_for_mem_enc or not eligibility[object_id]:
                    signature += (object_id,)
                groups.setdefault(signature, []).append(object_id)
            batches = [ids[n:n + self.decoder_batch_size] for ids in groups.values() for n in range(0, len(ids), self.decoder_batch_size)]
            for object_ids in batches:
                batch = len(object_ids)
                is_initial = object_ids[0] in boxes
                point_inputs = None
                if is_initial:
                    xy = torch.tensor([boxes[i] for i in object_ids], dtype=torch.float32, device=self.device).reshape(batch, 2, 2)
                    xy = xy / xy.new_tensor([w, h]) * self.image_size
                    point_inputs = {"point_coords": xy, "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=self.device).expand(batch, -1)}
                out = self.model.track_step(
                    frame_idx=frame_index, is_init_cond_frame=is_initial,
                    current_vision_feats=[f.expand(-1, batch, -1) for f in features],
                    current_vision_pos_embeds=[p.expand(-1, batch, -1) for p in positions],
                    feat_sizes=sizes, point_inputs=point_inputs, mask_inputs=None,
                    output_dict=self._merge_histories(object_ids, histories), num_frames=total_frames, run_mem_encoder=True,
                )
                scores = F.interpolate(out["pred_masks"].float(), (h, w), mode="bilinear", align_corners=False)
                masks = (scores[:, 0] > 0).cpu().numpy()
                visibility = out["object_score_logits"].float().sigmoid().flatten().cpu().tolist()
                for n, object_id in enumerate(object_ids):
                    results[object_id] = (masks[n], float(visibility[n]))
                    _, _, mw, mh = cv2.boundingRect(masks[n].view(np.uint8))
                    eligible = min(mw * self.image_size / w, mh * self.image_size / h) >= 16
                    compact = {}
                    for key in ["maskmem_features", "maskmem_pos_enc", "pred_masks", "obj_ptr", "object_score_logits"]:
                        value = out[key]
                        # Clone each slice so removing one track releases its GPU
                        # memory instead of retaining a whole old batch allocation.
                        if isinstance(value, list):
                            compact[key] = [v[n:n + 1] if batch == 1 else v[n:n + 1].clone() for v in value]
                        elif value is not None:
                            compact[key] = value[n:n + 1] if batch == 1 else value[n:n + 1].clone()
                        else:
                            compact[key] = None
                    if compact["maskmem_features"] is not None:
                        compact["maskmem_features"] = compact["maskmem_features"].to(torch.bfloat16 if self.device == "cuda" else torch.float32)
                    self.pending[object_id] = (frame_index, is_initial, compact, eligible)
        self.last_frame = frame_index
        return results


def render_frame(bgr, mask, label=""):
    result = bgr.copy()
    if mask.any():
        result[mask] = (result[mask].astype(np.float32) * 0.6 + np.array([80, 220, 80]) * 0.4).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (80, 240, 80), 2)
    if label:
        cv2.putText(result, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return result


PALETTE = [(80, 220, 80), (255, 160, 60), (80, 100, 255), (220, 80, 220),
           (40, 220, 240), (230, 210, 70), (170, 100, 240), (80, 170, 255)]


def render_objects(bgr, masks, label="", boxes=None):
    result = bgr.copy()
    for object_id, mask in masks.items():
        if not mask.any():
            continue
        color = PALETTE[(object_id - 1) % len(PALETTE)]
        if boxes is not None:
            x, y, x2, y2 = boxes[object_id]
        else:
            x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
            x2, y2 = x + width, y + height
        roi, local_mask = result[y:y2, x:x2], mask[y:y2, x:x2]
        if not roi.size or not local_mask.any():
            continue
        local_u8 = local_mask.view(np.uint8)
        tinted = cv2.addWeighted(roi, .6, np.full_like(roi, color), .4, 0)
        cv2.copyTo(tinted, local_u8, roi)
        contours, _ = cv2.findContours(local_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours, -1, color, 2)
        cv2.putText(result, f"ID {object_id}", (x, max(55, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if label:
        cv2.putText(result, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def render_preview(bgr, masks, label="", boxes=None, width=960):
    h, w = bgr.shape[:2]
    if w <= width:
        return render_objects(bgr, masks, label, boxes)
    height = round(h * width / w)
    preview = cv2.resize(bgr, (width, height))
    small_masks = {i: cv2.resize(m.view(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).view(bool)
                   for i, m in masks.items()}
    small_boxes = None if boxes is None else {i: [round(b[0] * width / w), round(b[1] * height / h),
                                                round(b[2] * width / w), round(b[3] * height / h)] for i, b in boxes.items()}
    return render_objects(preview, small_masks, label, small_boxes)


def jpeg_preview(bgr, width=1280, quality=82):
    if bgr.shape[1] > width:
        bgr = cv2.resize(bgr, (width, round(bgr.shape[0] * width / bgr.shape[1])))
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()
