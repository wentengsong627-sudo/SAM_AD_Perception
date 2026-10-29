"""Real A/D -> installed FlyCo Provider -> actual TrackingPipeline integration.

No ROS master or simulator is started. Synthetic point-cloud message I/O is
explicitly shimmed; the projector math is the original unmodified class body.
"""
import ast
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--flyco',type=Path,required=True)
_parser.add_argument('--video',type=Path,required=True)
_args=_parser.parse_args()
BUNDLE=ROOT
CLONE=_args.flyco.resolve()
PACKAGE=CLONE/'perception_predictor/src/sam_tracking/src/sam_tracking'
OUT=ROOT/'runtime/flyco-validation'
sys.path[:0]=[str(BUNDLE),str(PACKAGE.parent)]
from sam_ad.server import create_server
from sam_tracking.providers.factory import make_camera_provider
from sam_tracking.core import TrackingPipeline, PromptSession, DepthGuidancePolicy


def projection_check(mask):
    path=PACKAGE/'core/pointcloud_projector.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PointCloudProjector')
    shim=SimpleNamespace(read_points=lambda cloud,**kw:iter(cloud.points),
        create_cloud_xyz32=lambda header,points:SimpleNamespace(header=header,points=np.asarray(points)))
    scope={'np':np,'cv2':cv2,'copy':copy,'pc2':shim,'point_cloud2':shim}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
    positions=np.argwhere(mask>0); y,x=positions[len(positions)//2]
    camera=SimpleNamespace(project3dToPixel=lambda p:(100*p[0]/p[2]+int(x),100*p[1]/p[2]+int(y)))
    projector=scope['PointCloudProjector'](camera,np.eye(3),lambda p,t:p,lambda p:p)
    pose=np.eye(4);pose[:3,3]=[1,2,3]
    points=np.array([[0,0,8],[0,0,5],[0,0,-5],[1000,0,5],[-int(x)*.05,-int(y)*.05,5]])
    cloud=SimpleNamespace(header=SimpleNamespace(frame_id='world'),points=points+pose[:3,3])
    result=projector.project_mask_to_world_points([cloud],mask,None,None,(640,480),camera_pose=pose)
    assert result['selected_points_world'].shape==(1,3)
    assert np.allclose(result['selected_points_world'][0],[1,2,8])
    return {'original_projector_math_executed':True,'synthetic_points':5,'selected_points':1,
        'selected_world_xyz':[1,2,8],'ROS_message_IO':'explicit test shim; not ROS transport or real depth acceptance'}


def run(backend,frames):
    server=create_server(backend,root=BUNDLE,port=0)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05},daemon=True);thread.start()
    provider=None;executor=server.executor
    try:
        started=time.monotonic();last=started
        while not executor.ready.wait(.25):
            if time.monotonic()-started>180:raise TimeoutError('Model startup timed out')
            if time.monotonic()-last>8:
                print(json.dumps({'backend':backend,'phase':'loading'}),flush=True);last=time.monotonic()
        if executor.error:raise RuntimeError(str(executor.error))
        endpoint='http://127.0.0.1:'+str(server.server_address[1])
        provider=make_camera_provider(backend,None,endpoint)
        prompt=PromptSession();prompt.set_query_text('dog')
        pipeline=TrackingPipeline(provider,prompt,None,None)
        rows=[];observations=[];input_hash=hashlib.sha256();before=server.health()['process_calls']
        for index,rgb in enumerate(frames[:40]):
            provider.begin_frame(index)
            ok,reason,action,initial=pipeline.sync_provider_state(rgb);assert ok,reason
            tracked=pipeline.build_tracked_result_from_mask(rgb,initial,None) if initial is not None else pipeline.track_frame(rgb,None)
            mask=None if tracked is None else tracked.get('track_mask')
            meta=provider.last_metadata
            input_hash.update(rgb.tobytes())
            assert meta['input_rgb_sha256']==hashlib.sha256(rgb.tobytes()).hexdigest()
            if mask is not None:
                assert mask.shape==(480,640) and mask.dtype==np.uint8 and np.isin(mask,[0,255]).all() and mask.any()
                assert tracked['state']=='tracked'
                assert np.array_equal(tracked['mask_rgba'][:,:,3],mask)
                assert not tracked['masked_rgb'][mask==0].any()
                assert meta['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
                observations.append((index,mask.copy(),meta['ids']))
            else:assert not meta['ids']
            repeated=provider.track(rgb)
            assert (repeated is None)==(mask is None)
            if mask is not None:assert np.array_equal(repeated,mask)
            assert server.health()['process_calls']-before==index+1,'Callback predicted more than once'
            rows.append({'frame':index,'visible':mask is not None,'ids':meta['ids'],'states':meta['states'],
                'FlyCo_state':None if tracked is None else tracked['state'],'input_rgb_sha256':meta['input_rgb_sha256'],
                'mask_sha256':meta['mask_sha256'],'model_calls':server.health()['process_calls']-before})
            if index%10==0:print(json.dumps({'backend':backend,**rows[-1]}),flush=True)
        assert observations,'All masks were empty'
        assert all(ids==observations[0][2] for _,_,ids in observations),'Identity changed in integration smoke'
        rgb=frames[40];provider.begin_frame(40)
        depth=DepthGuidancePolicy(enabled=True,reinit_cooldown_frames=0,min_mask_area=32,empty_mask_streak_threshold=1)
        pipeline.depth_guidance=depth
        references=executor.submit(lambda host:{i:id(t['reference']) for i,t in host.pipe.tracks.items() if t.get('verified')}).result(10)
        old_count=depth.snapshot().reinit_count
        wrong=np.zeros((480,640),np.uint8);wrong[10:100,10:100]=255
        pending=pipeline.track_frame(rgb,None,depth_mask=wrong)
        assert pending['state']=='reinit_pending' and pending['track_mask'] is None
        same_references=executor.submit(lambda host:{i:id(t['reference']) for i,t in host.pipe.tracks.items() if t.get('verified')}).result(10)
        assert references==same_references and depth.snapshot().reinit_count==old_count
        pipeline.depth_guidance=None
        for index in (41,42,43):
            rgb=frames[index];provider.begin_frame(index);pipeline.track_frame(rgb,None)
        old_session=provider._session
        prompt.set_query_text('car');rgb=frames[44];provider.begin_frame(44)
        ok,reason,action,initial=pipeline.sync_provider_state(rgb)
        assert ok and action=='prompt_reinitialized' and provider._session!=old_session
        projection=projection_check(observations[-1][1])
        bgr=cv2.cvtColor(frames[observations[-1][0]],cv2.COLOR_RGB2BGR)
        mask=observations[-1][1]>0;preview=bgr.copy();preview[mask]=(preview[mask]*.6+np.array([245,171,50])*.4).astype(np.uint8)
        cv2.imwrite(str(OUT/(backend+'-FlyCo-visible.jpg')),preview)
        result={'backend':backend,'passed':True,'video_frames':40,'visible_mask_frames':len(observations),
            'observed_IDs':sorted({identity for _,_,ids in observations for identity in ids}),
            'input_sequence_sha256':input_hash.hexdigest(),'same_callback_one_model_prediction':True,
            'RGB_and_binary_mask_integrity':True,'original_RGBA_contract':True,'explicit_target_switch':True,
            'depth_hint_preserves_reference':True,'depth_hint_not_counted_as_recovery':True,
            'actual_FlyCo_pipeline':str(PACKAGE/'core/tracking_pipeline.py'),'model_source':str(BUNDLE),
            'synthetic_projection':projection,'scope':'Real CUDA model and actual FlyCo core; no ROS master, Unreal, mesh prediction or flight executed.'}
        (OUT/(backend+'-integration.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        (OUT/(backend+'-frames.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
        return result
    finally:
        if provider is not None:
            try:provider.reset()
            except Exception:pass
        server.shutdown();server.server_close();executor.close();thread.join(5)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(_args.video));frames=[]
    try:
        for _ in range(45):
            ok,bgr=capture.read();assert ok
            frames.append(cv2.cvtColor(cv2.resize(bgr,(640,480)),cv2.COLOR_BGR2RGB))
    finally:capture.release()
    results=[]
    for backend in ('A','D'):
        results.append(run(backend,frames));gc.collect()
        import torch
        torch.cuda.empty_cache()
    assert results[0]['input_sequence_sha256']==results[1]['input_sequence_sha256']
    installation={'note':'Installer hash/conflict checks are recorded separately in validation.json'}
    result={'passed':True,'results':results,'install_validation':installation,'same_input_frames':True,
        'ROS_runtime_executed':False,'Unreal_executed':False,'mesh_prediction_executed':False,'flight_executed':False,
        'remaining_acceptance':'Run original FlyCo ROS/3D prediction/planning system on the target host with actual aligned image, pose and point cloud inputs.'}
    (OUT/'integration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'A_frames':40,'D_frames':40,'protected_interfaces':'Unchanged; see installer validation'}),flush=True)


if __name__=='__main__':main()
