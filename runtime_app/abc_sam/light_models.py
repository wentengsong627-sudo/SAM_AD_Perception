"""EfficientSAM3 TinyViT D inference. Identity checks live in image_tracking."""
import sys
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from text_sam.vision import ROOT
from text_sam.targets import select_targets, associate, box_iou
from text_sam.identity import appearance
from text_sam.trust import similarity


class LightImageEngine:
    def __init__(self, scheme):
        sys.path.insert(0, str(ROOT / "third_party/efficientsam3"))
        sys.path.insert(0, str(ROOT / "third_party/efficientsam3/sam3"))
        import sam3.model_builder as builders
        from sam3.model_builder import build_efficientsam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        kwargs = dict(text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16, load_from_HF=False)
        if scheme != "D":
            raise ValueError("This release includes only EfficientSAM3 scheme D")
        path = ROOT / "models/efficientsam3/efficientsam3_tinyvit.pt"
        builder = build_efficientsam3_image_model
        kwargs.update(backbone_type="tinyvit", model_name="11m")
        if not path.exists():
            raise RuntimeError(f"方案 {scheme} 权重尚未下载完成：{path.name}")
        original_load=builders._load_checkpoint
        self.loading_info={}
        def checked_load(model, checkpoint_path):
            original_state_load=model.load_state_dict
            def capture(state_dict, *args, **kw):
                result=original_state_load(state_dict,*args,**kw)
                self.loading_info={"missing_keys":list(result.missing_keys),"unexpected_keys":list(result.unexpected_keys)}
                if result.missing_keys or result.unexpected_keys:
                    raise RuntimeError(f"Incomplete D checkpoint loading: {self.loading_info}")
                return result
            model.load_state_dict=capture
            try:
                original_load(model,checkpoint_path)
            finally:
                model.load_state_dict=original_state_load
        builders._load_checkpoint=checked_load
        try:
            self.model = builder(checkpoint_path=str(path), **kwargs).eval()
        finally:
            builders._load_checkpoint=original_load
        self.processor = Sam3Processor(self.model)
        self.scheme = scheme
        self.calls = 0
        self.last_committed_ids = []
        self.actual_size = getattr(self.processor,"resolution",1008)
        self.text_cache={}
        self.text_encoder_calls=0
        self.cache_enabled=os.environ.get('ABCD_RUNTIME_CACHE','1')=='1'
        if self.cache_enabled:
            from .runtime_cache import cache_encoder_metadata
            self.native_prepare=cache_encoder_metadata(self.model.transformer.encoder)

    @torch.inference_mode()
    def segment(self, rgb, text, threshold):
        text=text.rstrip('.')
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))
            self.processor.confidence_threshold = threshold
            if text not in self.text_cache or not self.cache_enabled:
                tokenizer=self.model.backbone.language_backbone.tokenizer
                if len(tokenizer.encode(text))+2>16:
                    raise ValueError("文字描述超过 ctx16 编码器的 14 个内容 token；请缩短描述。请使用简短的英文目标描述。")
                if len(self.text_cache)>=8:self.text_cache.clear()
                self.text_cache[text]=self.model.backbone.forward_text([text],device=self.processor.device)
                self.text_encoder_calls+=1
            state['backbone_out'].update(self.text_cache[text])
            state['geometric_prompt']=self.model._get_dummy_prompt()
            state=self.processor._forward_grounding(state)
        self.calls += 1
        masks = state["masks"].detach().cpu().numpy().reshape(-1,*rgb.shape[:2])
        scores = state["scores"].float().detach().cpu().tolist()
        boxes_tensor=state["boxes"].float().detach().clone()
        boxes_tensor[:,0::2].clamp_(0,rgb.shape[1])
        boxes_tensor[:,1::2].clamp_(0,rgb.shape[0])
        boxes = boxes_tensor.cpu().tolist()
        return masks, boxes, scores
