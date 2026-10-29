"""D image masks with transactional application identity validation.

Shares A's identity lifecycle, never SAM2 video features or a new neural model.
Only accepted boxes/descriptors are committed; image inference is stateless.
"""
import time
import cv2
import numpy as np
from text_sam.targets import select_targets,associate,box_iou
from text_sam.identity import appearance
from text_sam.trust import similarity
from text_sam.pipeline import TrackingPipeline


class ImageGrounder:
    search_every_frame=True
    defer_suspect_search=True

    def __init__(self,engine,threshold):
        self.engine,self.threshold=engine,threshold
        self.rgb=self.command=None
        self.data=None;self.calls=0;self.frame_index=None;self.sample_key=None

    def sample(self,rgb):
        key=(self.frame_index,self.command.query)
        if self.rgb is rgb and key==self.sample_key:return self.data
        masks,boxes,scores=self.engine.segment(rgb,self.command.query,self.threshold)
        order=sorted(range(len(boxes)),key=lambda i:scores[i],reverse=True);keep=[]
        for n in order:
            if not masks[n].any():continue
            duplicate=False
            for k in keep:
                if box_iou(boxes[n],boxes[k])<.5:continue
                x=min(int(boxes[n][0]),int(boxes[k][0]));y=min(int(boxes[n][1]),int(boxes[k][1]))
                a=max(int(boxes[n][2])+1,int(boxes[k][2])+1);b=max(int(boxes[n][3])+1,int(boxes[k][3])+1)
                left,right=masks[n][y:b,x:a],masks[k][y:b,x:a]
                union=np.count_nonzero(left|right)
                if union and np.count_nonzero(left&right)/union>.75:duplicate=True;break
            if not duplicate:keep.append(n)
        self.rgb=rgb;self.sample_key=key
        self.data=([masks[n] for n in keep],[boxes[n] for n in keep],[scores[n] for n in keep])
        return self.data

    def locate(self,rgb,command,threshold=.3):
        started=time.perf_counter();self.command=command
        masks,boxes,scores=self.sample(rgb)
        result=select_targets(boxes,scores,command,rgb.shape[1],rgb.shape[0],threshold)
        result['grounder']='EfficientSAM3 TinyViT11M + guarded identity recovery'
        result['grounding_ms']=(time.perf_counter()-started)*1000
        self.calls+=1
        return result


class ImageMaskTracker:
    prediction_source='efficientsam3'
    has_neural_memory=False

    def __init__(self,grounder):
        self.grounder=grounder;self.min_visibility=grounder.threshold;self.reset()

    def reset(self):
        self.histories={};self.pending={};self.last_committed_ids=[];self.last_rejected_ids=[]
        self.grounder.engine.identity_commits=0

    @property
    def history_frames(self):return 0

    def drop_objects(self,ids):
        for i in ids:self.histories.pop(i,None);self.pending.pop(i,None)

    def predict_many(self,rgb,frame_index,total_frames,boxes=None):
        masks,candidates,scores=self.grounder.sample(rgb);boxes=boxes or {}
        self.pending={};outputs={};used=set();h,w=rgb.shape[:2]
        chosen={}
        # Verification prompts come from the lifecycle's category/identity search.
        for i,box in boxes.items():
            available=[n for n in range(len(candidates)) if n not in used]
            n=max(available,key=lambda n:box_iou(box,candidates[n]),default=None)
            if n is not None and box_iou(box,candidates[n])>=.2:chosen[i]=n;used.add(n)
        remaining=[n for n in range(len(candidates)) if n not in used]
        stable={i:t['box'] for i,t in self.histories.items() if i not in boxes}
        descriptors={n:appearance(rgb,candidates[n],masks[n]) for n in remaining}
        affinity=[[box_iou(box,candidates[n]) if similarity(self.histories[i]['appearance'],descriptors[n])>=.4 else 0.
                   for n in remaining] for i,box in stable.items()]
        # In an unresolved crossing, wait rather than force a spatial identity swap.
        for row in affinity:
            values=sorted(row,reverse=True)
            if len(values)>1 and values[0]>=.15 and values[0]-values[1]<.035:
                row[:]=[0.]*len(row)
        matches,_=associate(stable,[{'box':candidates[n]} for n in remaining],.15,affinity)
        chosen.update({i:remaining[n] for i,n in matches.items()})
        for i in set(self.histories)|set(boxes):
            if i not in chosen:outputs[i]=(np.zeros((h,w),bool),0.);continue
            n=chosen[i];mask=masks[n];x,y,a,b=cv2.boundingRect(mask.view(np.uint8))
            self.pending[i]={'box':[x,y,x+a,y+b],'appearance':appearance(rgb,[x,y,x+a,y+b],mask)}
            outputs[i]=(mask,float(scores[n]))
        return outputs

    def commit_many(self,ids):
        accepted=set(ids);self.last_committed_ids=sorted(accepted&self.pending.keys())
        self.last_rejected_ids=sorted(self.pending.keys()-accepted)
        for i in self.last_committed_ids:self.histories[i]=self.pending[i]
        self.grounder.engine.identity_commits+=len(self.last_committed_ids)
        self.pending={}


class GuardedImagePipeline(TrackingPipeline):
    def __init__(self,grounder,tracker,threshold=.3,interval=30,max_objects=8,occlusion=90,stride=1,discover=True):
        locator=ImageGrounder(tracker,threshold)
        super().__init__(locator,ImageMaskTracker(locator),threshold,interval,max_objects,occlusion,1,discover)

    def process(self,rgb,frame_index,total_frames,command=None):
        self.grounder.frame_index=frame_index
        return super().process(rgb,frame_index,total_frames,command)
