"""Lightweight evidence gates, never a semantic classifier or learned ReID."""
import cv2
import numpy as np
from .identity import appearance
from .targets import box_iou


def observe(rgb, mask, visibility):
    if not mask.any() or not np.isfinite(visibility):
        return None
    x,y,w,h=cv2.boundingRect(mask.view(np.uint8))
    return {'box':[x,y,x+w,y+h],'area':int(np.count_nonzero(mask)),
            'aspect':w/max(1,h),'appearance':appearance(rgb,[x,y,x+w,y+h],mask),
            'visibility':float(visibility)}


def similarity(a,b):
    if a is None or b is None:return 0.
    # Most identity evidence is hue/saturation; brightness receives lower weight
    # so a cast shadow does not by itself replace an otherwise matching identity.
    ca=np.sqrt((a.reshape(12,4,4)**2).sum(axis=2)).ravel()
    cb=np.sqrt((b.reshape(12,4,4)**2).sum(axis=2)).ravel()
    return float(.75*np.dot(ca,cb)+.25*np.dot(a,b))


def validate(track, observation, source, recovery=False):
    if observation is None or observation['visibility']<track.get('min_visibility',.5):
        return False,{'reason':'empty_or_invisible','confidence':0.}
    if track.get('state')=='VERIFYING':
        box=track['candidate_box'];size=max(1,(box[2]-box[0])*(box[3]-box[1]))
        fill=observation['area']/size
        overlap=box_iou(box,observation['box'])
        if overlap<.2 or not .04<=fill<=1.5:
            return False,{'reason':'prompt_mask_disagreement','confidence':0.,'box_iou':overlap,'mask_fill':fill}
    ref=track.get('reference') or track.get('provisional')
    if ref is None:
        return observation['area']>=4,{'reason':'bootstrap','confidence':observation['visibility']}
    online=track.get('online') or ref
    anchor=similarity(ref['appearance'],observation['appearance'])
    current=similarity(online['appearance'],observation['appearance'])
    last=track.get('trusted_observation') or ref
    area=observation['area']/max(1,last['area'])
    aspect=observation['aspect']/max(.01,last['aspect'])
    fixed_area=observation['area']/max(1,ref['area'])
    a,b=last['box'],observation['box']
    distance=np.hypot((b[0]+b[2]-a[0]-a[2])/2,(b[1]+b[3]-a[1]-a[3])/2)/max(8,np.hypot(a[2]-a[0],a[3]-a[1]))
    reasons=[]
    if anchor < (.60 if recovery else .43):reasons.append('fixed_appearance')
    if current < (.54 if recovery else .40):reasons.append('online_appearance')
    if recovery:
        if not .12<=fixed_area<=8:reasons.append('identity_scale')
    else:
        if not .16<=area<=4:reasons.append('area_jump')
        if not .22<=aspect<=4.5:reasons.append('aspect_jump')
        if distance>1.8 and anchor<.8:reasons.append('motion_jump')
    if source=='motion' and (anchor<.52 or current<.50):reasons.append('unverified_propagation')
    confidence=float(.45*anchor+.30*current+.25*observation['visibility'])
    return not reasons,{'reason':','.join(reasons) or 'accepted','anchor_similarity':anchor,
        'online_similarity':current,'area_ratio':area,'fixed_area_ratio':fixed_area,
        'aspect_ratio':aspect,'motion_distance':float(distance),'confidence':confidence,
        'fresh_visibility':source!='motion'}


def fixed_reference(observation, rgb):
    ref=dict(observation)
    ref['appearance']=observation['appearance'].copy()
    ref['appearance'].setflags(write=False)
    ref['crop_appearance']=appearance(rgb,ref['box'])
    return ref


def update_online(track, observation):
    previous=track['online']['appearance']
    vector=previous*.97+observation['appearance']*.03
    vector/=max(1e-8,np.linalg.norm(vector))
    track['online']={**observation,'appearance':vector}
