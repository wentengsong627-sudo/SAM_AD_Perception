"""Independent identity lifecycles with transactional SAM memory acceptance."""
import cv2
import time
import numpy as np
from .targets import associate,box_iou
from .identity import appearance,update_motion
from .motion import MotionPropagation
from .trust import observe,validate,similarity,fixed_reference,update_online

TRACKING,SUSPECT,OCCLUDED,LOST,REACQUIRING,VERIFYING=('TRACKING','SUSPECT','OCCLUDED','LOST','REACQUIRING','VERIFYING')


class TrackingPipeline:
    def __init__(self,grounder,tracker,threshold=.30,redetect_interval=30,max_objects=8,occlusion_frames=90,segmentation_stride=1,discover_new=True):
        self.grounder,self.tracker=grounder,tracker
        self.threshold,self.redetect_interval,self.max_objects=threshold,redetect_interval,max_objects
        self.occlusion_frames,self.segmentation_stride=occlusion_frames,segmentation_stride
        self.discover_new=discover_new
        self.search_gate_enabled=getattr(grounder,'recovery_gate_enabled',False)
        self.partial_retry_enabled=getattr(grounder,'recovery_partial_retry',False)
        self.partial_retry_searches=0
        self.search_gate_skips=0;self.search_gate_ms=0.;self.search_gate_reasons={}
        self.tracks={};self.next_id=1;self.command=None
        self.detection_count=0;self.neural_frames=0;self.propagated_frames=0
        self.sam_encoder_calls=0;self.memory_updates=0;self.rejected_predictions=0
        self.neural_index=0;self.last_neural_frame=None;self.last_detection_frame=None
        self.previous_masks={};self.previous_visibility={}
        self.motion=MotionPropagation() if segmentation_stride>1 else None
        self.audit=[];self.transitions=[];self.seen_any=False
        self.tracker.reset()

    def state(self,track,state,frame,reason):
        if track['state']!=state:
            self.transitions.append({'id':track['id'],'frame':frame,'from':track['state'],'to':state,'reason':reason})
            track['state']=state

    def public(self,track,frame):
        keys=['id','box','state','empty_streak','visibility','first_frame','last_seen','grounding_score','verified','confidence']
        row={k:track.get(k) for k in keys}
        row['observed_box']=track.get('observed_box')
        row['prediction_only']=track['state']!=TRACKING
        if row['prediction_only']:
            gap=min(10,max(0,frame-track['motion_frame']))
            dx,dy=np.asarray(track['velocity'])*gap
            a,b,c,d=track['motion_box'];row['predicted_box']=[float(a+dx),float(b+dy),float(c+dx),float(d+dy)]
        row['reference_frame']=track.get('reference_frame')
        return row

    def candidate_affinity(self,track,item,descriptor,frame):
        ref=track.get('reference')
        if ref is None:return 0.
        app=max(similarity(ref['appearance'],descriptor),similarity(ref.get('crop_appearance'),descriptor))
        box=item['box'];old=track['box']
        scale=max(8,np.hypot(old[2]-old[0],old[3]-old[1]))
        gap=max(0,frame-track['last_seen'])
        step=min(10,gap);dx,dy=np.asarray(track['velocity'])*step
        predicted=[old[0]+dx,old[1]+dy,old[2]+dx,old[3]+dy]
        distance=float(np.hypot((box[0]+box[2]-predicted[0]-predicted[2])/2,(box[1]+box[3]-predicted[1]-predicted[3])/2)/scale)
        if app<.38:return 0.
        if gap<15 and distance>2:return 0.
        size=(box[2]-box[0])*(box[3]-box[1])/max(1,(ref['box'][2]-ref['box'][0])*(ref['box'][3]-ref['box'][1]))
        if not .08<=size<=12:return 0.
        return .75*app+.15*max(0,1-distance/4)+.1*min(size,1/max(size,.001))

    def process(self,rgb,frame_index,total_frames,command=None):
        self.audit=[];self.transitions=[]
        switched=command is not None
        if switched:
            self.command=command;self.tracks={};self.tracker.reset();self.seen_any=False
            self.last_detection_frame=None;self.last_neural_frame=None;self.neural_index=0
            self.previous_masks={};self.previous_visibility={}
            if self.motion is not None:self.motion=MotionPropagation()
            context=getattr(self.tracker,'set_task_context',None)
            if context:context(self.command.quantity)
        if self.command is None:raise RuntimeError('请先指定目标。')
        added,recovered,retired,warnings=[],[],[],[]
        new_boxes={};detection=None
        for t in self.tracks.values():
            if t['state']==OCCLUDED and frame_index-t['last_seen']>self.occlusion_frames:
                self.state(t,LOST,frame_index,'occlusion_timeout_reference_retained')
        missing=[t for t in self.tracks.values() if t['state'] in {OCCLUDED,LOST} or
                 (t['state']==SUSPECT and not getattr(self.grounder,'defer_suspect_search',False))]
        active=[t for t in self.tracks.values() if t['state']==TRACKING]
        verifying=any(t['state']==VERIFYING for t in self.tracks.values())
        search=bool(missing) or (not self.seen_any and not verifying) or (self.command.quantity=='all' and self.discover_new)
        interval=min(12,self.redetect_interval) if missing and not active else self.redetect_interval
        if getattr(self.grounder,'search_every_frame',False):interval=1
        due=search and (self.last_detection_frame is None or frame_index-self.last_detection_frame>=interval)
        fast_due=self.partial_retry_enabled and self.command.quantity=='one' and not active and any(t.get('retry_search_frame',float('inf'))<=frame_index for t in missing)
        due=due or fast_due
        pass  # Board-only path omitted from this CUDA release.
        if due:
            if fast_due:
                self.partial_retry_searches+=1
                self.audit.append({'action':'expedite_recovery_search','reason':'partial_boundary_candidate','frame':frame_index})
            context=getattr(self.grounder,'set_task_context',None)
            if context:context(self.seen_any,self.command.quantity)
            locate=getattr(self.grounder,'locate_guarded',self.grounder.locate)
            detection=locate(rgb,self.command,self.threshold)
            self.detection_count+=1;self.last_detection_frame=frame_index
            for t in missing:
                t.pop('retry_search_frame',None)
                if t.get('search_gate') is not None:t['search_gate'].mark_searched(rgb,frame_index)
            candidates=detection['candidates'] if self.seen_any and self.command.quantity=='one' else detection['selected']
            accepted_candidates=[]
            for item in candidates:
                if item.get('category_ok',True):accepted_candidates.append(item)
                else:self.audit.append({'candidate':item['id'],'action':'reject_candidate','reason':'category_contradiction','evidence':item.get('category_evidence')})
            descriptors=[appearance(rgb,c['box']) for c in accepted_candidates]
            used=set();candidate_tracks={}
            # Discovery must not replace an already trusted mask with a box.
            for n,item in enumerate(accepted_candidates):
                overlaps=[t for t in self.tracks.values() if t['state'] in {TRACKING,VERIFYING} and box_iou(t['box'],item['box'])>.35]
                if overlaps:
                    t=max(overlaps,key=lambda t:box_iou(t['box'],item['box']))
                    used.add(n);candidate_tracks[item['id']]=t['id']
            available=[n for n in range(len(accepted_candidates)) if n not in used]
            restore=[t for t in self.tracks.values() if t['verified'] and t['state'] in {SUSPECT,OCCLUDED,LOST}]
            proposals=[accepted_candidates[n] for n in available]
            affinity=[[self.candidate_affinity(t,item,descriptors[available[j]],frame_index) for j,item in enumerate(proposals)] for t in restore]
            for n,row in enumerate(affinity):
                best=sorted(row,reverse=True)
                if len(best)>1 and best[0]>.5 and best[0]-best[1]<.035:
                    affinity[n]=[0.]*len(row)
                    used.update(available[j] for j,w in enumerate(row) if w>.5)
                    self.audit.append({'id':restore[n]['id'],'action':'reject_recovery','reason':'ambiguous_candidates'})
            # If two identities explain one candidate equally well, wait.
            for j in range(len(proposals)):
                scores=sorted([row[j] for row in affinity],reverse=True)
                if len(scores)>1 and scores[0]>.5 and scores[0]-scores[1]<.035:
                    for row in affinity:row[j]=0.
                    used.add(available[j])
                    self.audit.append({'candidate':proposals[j]['id'],'action':'reject_candidate','reason':'ambiguous_identity'})
            matches,_=associate({t['id']:t['box'] for t in restore},proposals,.43,affinity)
            slots=self.max_objects-sum(t['state'] in {TRACKING,SUSPECT,VERIFYING} for t in self.tracks.values())
            for t in restore:
                if t['id'] not in matches:continue
                if t['state']!=SUSPECT and slots<=0:continue
                if t['state']!=SUSPECT:slots-=1
                n=available[matches[t['id']]];item=accepted_candidates[n]
                self.state(t,REACQUIRING,frame_index,'category_and_coarse_identity_passed')
                self.state(t,VERIFYING,frame_index,'masked_identity_verification')
                t['verify_count']=0;t['recovering']=True;t['candidate_box']=item['box'];t['grounding_score']=item['score']
                new_boxes[t['id']]=item['box'];used.add(n);candidate_tracks[item['id']]=t['id']
            allow_new=not self.seen_any or (self.command.quantity=='all' and self.discover_new)
            capacity=self.max_objects-sum(t['state'] in {TRACKING,SUSPECT,REACQUIRING,VERIFYING} for t in self.tracks.values())
            for n,item in enumerate(accepted_candidates):
                if n in used or not allow_new:continue
                if any(t['state']==VERIFYING and box_iou(t['candidate_box'],item['box'])>.65 for t in self.tracks.values()):continue
                if capacity<=0:
                    warnings.append('达到同时跟踪上限，候选尚未加入。');continue
                object_id=self.next_id;self.next_id+=1;capacity-=1
                t={'id':object_id,'box':item['box'],'state':REACQUIRING,'empty_streak':0,'visibility':0.,
                   'grounding_score':item['score'],'verified':False,'confidence':0.,'verify_count':0,'recovering':False,
                   'first_frame':frame_index,'last_seen':frame_index,'motion_box':item['box'],'motion_frame':frame_index,
                   'velocity':[0.,0.],'candidate_box':item['box'],'trusted_observation':None,
                   'min_visibility':getattr(self.tracker,'min_visibility',.5)}
                self.tracks[object_id]=t;self.state(t,VERIFYING,frame_index,'new_instance_admission')
                new_boxes[object_id]=item['box'];candidate_tracks[item['id']]=object_id
                if self.command.quantity=='one':allow_new=False
            detection['candidate_track_ids']=candidate_tracks
            detection['untracked_count']=sum(n not in used and not allow_new for n in range(len(accepted_candidates)))
        for t in self.tracks.values():
            if t['state']==VERIFYING:new_boxes[t['id']]=t['candidate_box']
        force=any(t['state'] in {SUSPECT,VERIFYING} for t in self.tracks.values())
        update=(self.segmentation_stride==1 or switched or new_boxes or force or self.last_neural_frame is None or frame_index-self.last_neural_frame>=self.segmentation_stride)
        if update:
            neural_index=frame_index if self.segmentation_stride==1 else self.neural_index
            outputs=self.tracker.predict_many(rgb,neural_index,total_frames,new_boxes)
            source=getattr(self.tracker,'prediction_source','sam2')
            if source=='sam2':self.sam_encoder_calls+=bool(outputs)
            self.neural_index+=1;self.last_neural_frame=frame_index;self.neural_frames+=1
        else:
            retained={i:m for i,m in self.previous_masks.items() if i in self.tracks and self.tracks[i]['state']==TRACKING}
            outputs={i:(m,self.previous_visibility[i]) for i,m in self.motion.propagate(rgb,retained).items()}
            self.propagated_frames+=1;source='motion'
        masks={};commit=[];observations={}
        for object_id,(mask,visibility) in outputs.items():
            t=self.tracks[object_id];obs=observe(rgb,mask,visibility)
            valid,evidence=validate(t,obs,source,t.get('recovering',False))
            t['visibility']=float(visibility);t['confidence']=evidence['confidence']
            self.audit.append({'id':object_id,'action':'accept_prediction' if valid else 'reject_prediction','source':source,**evidence})
            if not valid:
                # A partial boundary candidate can justify an earlier new search,
                # never a mask output, memory commit, or looser identity gate.
                cb=t.get('candidate_box');h,w=rgb.shape[:2]
                boundary=cb is not None and (cb[0]<=1 or cb[1]<=1 or cb[2]>=w-1 or cb[3]>=h-1)
                if (self.partial_retry_enabled and self.command.quantity=='one' and t['verified'] and
                    t.get('recovering') and source!='motion' and boundary and
                    evidence.get('reason')=='identity_scale' and evidence.get('fixed_area_ratio',1)<.12 and
                    evidence.get('anchor_similarity',0)>=.85 and evidence.get('online_similarity',0)>=.80 and
                    t.get('partial_retries',0)<2):
                    t['partial_retries']=t.get('partial_retries',0)+1;t['retry_search_frame']=frame_index+3
                    self.audit.append({'id':object_id,'action':'schedule_recovery_recheck','reason':'partial_boundary_candidate','due_frame':frame_index+3,'attempt':t['partial_retries']})
                self.rejected_predictions+=1
                t['empty_streak']+=1
                target=OCCLUDED if t['state']==VERIFYING or t['empty_streak']>=2 else SUSPECT
                self.state(t,target,frame_index,evidence['reason'])
                if target==OCCLUDED:self.tracker.drop_objects([object_id])
                if not t['verified']:
                    # An unverified proposal has no identity to retain.
                    retired.append(object_id)
                continue
            t['empty_streak']=0
            if t['state']==VERIFYING:
                t['verify_count']+=1;t['candidate_box']=obs['box']
                if 'provisional' not in t and not t['verified']:t['provisional']=fixed_reference(obs,rgb)
                if t['verify_count']<2:continue
                if not t['verified']:
                    t['reference']=t['provisional'];t['online']=dict(t['reference']);t['reference_frame']=t['first_frame']
                    t['verified']=True;added.append(object_id);self.seen_any=True
                    pass  # Board-only path omitted from this CUDA release.
                else:recovered.append(object_id)
                self.state(t,TRACKING,frame_index,'verification_passed')
                t['recovering']=False
                t['partial_retries']=0;t.pop('retry_search_frame',None)
            elif t['state']==SUSPECT:
                self.state(t,VERIFYING,frame_index,'fresh_prediction_revalidated')
                self.state(t,TRACKING,frame_index,'trusted_history_prediction_passed')
            t['box']=obs['box'];masks[object_id]=mask
            observations[object_id]=(obs,evidence)
            if source!='motion':
                commit.append(object_id)
        # Resolve duplicate masks before committing either hypothesis to memory.
        ids=list(masks)
        for n,a in enumerate(ids):
            for b in ids[n+1:]:
                if a not in masks or b not in masks or box_iou(self.tracks[a]['box'],self.tracks[b]['box'])<.6:continue
                overlap=np.count_nonzero(masks[a]&masks[b]);union=np.count_nonzero(masks[a]|masks[b])
                if union and overlap/union>.75:
                    loser=b if self.tracks[a]['confidence']>=self.tracks[b]['confidence'] else a
                    masks.pop(loser);commit=[i for i in commit if i!=loser]
                    self.state(self.tracks[loser],SUSPECT,frame_index,'duplicate_mask')
                    if loser in added:
                        added.remove(loser);retired.append(loser)
        templates_updated=[]
        for object_id in commit:
            t=self.tracks[object_id];obs,evidence=observations[object_id]
            t['last_seen']=frame_index;t['observed_box']=obs['box'];t['trusted_observation']=obs
            update_motion(t,obs['box'],frame_index)
            crowded=any(i!=object_id and box_iou(t['box'],self.tracks[i]['box'])>.1 for i in masks)
            if not crowded and frame_index-t['first_frame']>5 and evidence.get('anchor_similarity',1)>.75:
                update_online(t,obs)
                templates_updated.append(object_id)
        if update:self.tracker.commit_many(commit)
        neural_memory=getattr(self.tracker,'has_neural_memory',True)
        if neural_memory:self.memory_updates+=len(commit)
        for event in self.audit:
            if 'id' in event:
                event['reliable_output']=event['id'] in masks
                event['memory_commit']=neural_memory and event['id'] in commit
                event['identity_commit']=event['id'] in commit
                event['template_update']=event['id'] in templates_updated
        for i in retired:
            self.tracks.pop(i,None);self.tracker.drop_objects([i])
        if self.motion is not None:
            self.motion.remember(rgb);self.previous_masks=masks
            self.previous_visibility={i:self.tracks[i]['visibility'] for i in masks}
        # Identity references survive release of neural history. Bound multi-object archives.
        archives=[i for i,t in self.tracks.items() if t['state']==LOST]
        if self.command.quantity=='all':
            for i in archives[:-128]:self.tracks.pop(i)
        phase='tracking' if masks else 'lost' if self.seen_any else 'searching'
        message=f'可靠可见目标 {len(masks)} 个' if masks else ('遮挡 / 丢失：无可靠可见掩码，正在重捕获原目标' if self.seen_any else '尚无通过验证的目标，继续查找')
        return {'masks':masks,'objects':[self.public(t,frame_index) for t in self.tracks.values()],
                'phase':phase,'message':message,'detection':detection,'added_ids':added,'retired_ids':retired,
                'recovered_ids':recovered,'warnings':warnings,'active_count':sum(t['state']==TRACKING for t in self.tracks.values()),
                'visible_count':len(masks),'history_frames':self.tracker.history_frames,'switched':switched,
                'segmentation_source':source,'audit':self.audit,'transitions':self.transitions}
