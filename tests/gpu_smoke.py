"""Real CUDA wire smoke test of the relocated portable bundle, sequential A then D."""
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT if (ROOT/'sam_ad').is_dir() else ROOT/'outputs/handoff/SAM_AD_Module_20261027'
OUT=BUNDLE/'runtime/validation'
sys.path.insert(0,str(BUNDLE))
from sam_ad import APIError, Client


def run(backend,frames):
    with socket.socket() as available:
        available.bind(('127.0.0.1',0)); port=available.getsockname()[1]
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Test port is occupied; leaving existing service untouched')
    token=secrets.token_urlsafe(32)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',SAM_AD_ADMIN_TOKEN=token,
             HF_HOME=str(BUNDLE/'runtime/hf-cache'),TORCH_HOME=str(BUNDLE/'runtime/torch-cache'),
             CUDA_CACHE_PATH=str(BUNDLE/'runtime/cuda-cache'))
    for name in ('HF_HOME','TORCH_HOME','CUDA_CACHE_PATH'):Path(env[name]).mkdir(parents=True,exist_ok=True)
    with (OUT/(backend+'-server.log')).open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-B','-u','-m','sam_ad.server','--backend',backend,'--port',str(port)],cwd=BUNDLE,env=env,
             stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        endpoint='http://127.0.0.1:'+str(port)
        client=Client(endpoint,backend=backend,timeout=60)
        try:
            start=time.monotonic(); last=start
            while True:
                if proc.poll() is not None:raise RuntimeError('Server exited: '+str(OUT/(backend+'-server.log')))
                try:
                    health=client.health()
                    if health['status']=='error':raise RuntimeError(health['error'])
                    if health['status']=='ready':break
                except (ConnectionError,OSError):pass
                except Exception as exc:
                    from sam_ad import TransportError
                    if not isinstance(exc,TransportError):raise
                if time.monotonic()-start>180:raise TimeoutError('Model did not become ready')
                if time.monotonic()-last>8:
                    print(json.dumps({'backend':backend,'phase':'loading','seconds':round(time.monotonic()-start,1)}),flush=True); last=time.monotonic()
                time.sleep(.25)
            assert health['warmup_identity_memory_cleared']
            assert Path(health['workspace']).resolve()==BUNDLE.resolve()
            assert health['loaded_modules']
            for path in health['loaded_modules'].values():Path(path).resolve().relative_to(BUNDLE.resolve())
            # Windows venv python.exe is a launcher; the actual worker is its child.
            import psutil
            assert health['pid']==proc.pid or proc.pid in [p.pid for p in psutil.Process(health['pid']).parents()]
            ready_seconds=time.monotonic()-start
            client.start('dog')
            rows=[]; times=[]; visible=0; ids=set(); hashes=[]; saved_preview=False
            for index,bgr in enumerate(frames):
                timestamp=index*33333333
                tick=time.perf_counter()
                result=client.process_bgr(bgr,timestamp_ns=timestamp)
                mask=result.mask
                times.append((time.perf_counter()-tick)*1000)
                assert result.data['sequence']==index
                assert result.data['backend']==backend
                assert result.data['image']=={'width':640,'height':480,'color_order':'RGB'}
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                assert result.data['input_rgb_sha256']==hashlib.sha256(rgb.tobytes()).hexdigest()
                hashes.append(result.data['input_rgb_sha256'])
                if result.visible:
                    assert mask is not None and mask.shape==(480,640) and np.count_nonzero(mask)>0
                    x,y,w,h=cv2.boundingRect(mask); assert result.bbox_xyxy==[x,y,x+w,y+h]
                    visible+=1; ids.update(result.data['ids'])
                    if not saved_preview:
                        preview=bgr.copy(); pixels=mask>0
                        preview[pixels]=(preview[pixels]*.6+np.array([245,171,50])*.4).astype(np.uint8)
                        cv2.imwrite(str(OUT/(backend+'-first-visible.jpg')),preview); saved_preview=True
                else:
                    assert result.mask is None and result.bbox_xyxy is None and result.data['ids']==[]
                rows.append(result.data)
                if index==4:
                    ok,encoded=cv2.imencode('.png',bgr); assert ok
                    payload={'session':client.session,'sequence':index,'timestamp_ns':timestamp,'image_png':base64.b64encode(encoded).decode()}
                    duplicate=client._request('/frame',payload)
                    assert duplicate==result.data
                    assert client.health()['process_calls']==index+1
                    wrong=dict(payload,sequence=index+2)
                    try:client._request('/frame',wrong); raise AssertionError('Out of order accepted')
                    except APIError as exc:assert exc.code=='OUT_OF_ORDER'
                    assert client.health()['process_calls']==index+1
                if index==20:
                    reference=client.hint('integration_smoke_test')
                    assert reference['backend']==backend
                if index%10==0:print(json.dumps({'backend':backend,'frame':index,'visible':result.visible,'states':result.states,'roundtrip_ms':round(times[-1],1)}),flush=True)
            assert visible>0,'All masks were empty; smoke test is not accepted'
            assert len(ids)==1,'Same selected dog changed ID during this smoke test'
            after=client.health(); assert after['process_calls']==len(frames)
            old=client.session; client.start('car'); assert client.session!=old
            client.reset(); assert not client.health()['busy']
            with (OUT/(backend+'-events.jsonl')).open('w',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
            return {'backend':backend,'ready_seconds':ready_seconds,'frames':len(frames),'visible_frames':visible,'output_ids':sorted(ids),
                'same_frame_duplicate_did_not_advance':True,'out_of_order_rejected':True,'hint_and_explicit_target_switch_passed':True,'session_released':True,
                'input_sequence_sha256':hashlib.sha256(''.join(hashes).encode()).hexdigest(),
                'client_roundtrip_ms':{'mean':float(np.mean(times)),'P95':float(np.percentile(times,95)),'max':float(max(times))},
                'health':after,'scope':'Integration smoke; no manually labeled accuracy/identity ground truth, no display, not a ranking benchmark'}
        finally:
            try:client.close()
            except Exception:pass
            try:
                req=urllib.request.Request(endpoint+'/_admin/shutdown',json.dumps({'token':token}).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=10) as response:response.read()
                proc.wait(timeout=15)
            except Exception:
                if proc.poll() is None:proc.terminate(); proc.wait(timeout=10)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True,help='Video containing the selected dog; 40 frames required')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    frames=[]; cap=cv2.VideoCapture(str(args.video))
    try:
        for _ in range(40):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError('Sample video shorter than expected')
            frames.append(cv2.resize(bgr,(640,480)))
    finally:cap.release()
    rows=[]
    for backend in ('A','D'):
        row=run(backend,frames); rows.append(row)
        (OUT/(backend+'-gpu-smoke.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
    assert rows[0]['input_sequence_sha256']==rows[1]['input_sequence_sha256']
    result={'passed':True,'platform':'Windows NVIDIA RTX 5060 Laptop 8GB','same_input_pixels':True,'results':rows,
        'clean_virtualenv_install_executed':False,'Linux_GPU_executed':False,'weights_and_source_loaded_from_export':True,
        'note':'Both services executed sequentially using the existing tested Python dependency environment; no CPU fallback, no Unreal execution.'}
    (OUT/'gpu-smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'results':[{'backend':r['backend'],'visible_frames':r['visible_frames'],'frames':r['frames']} for r in rows]}),flush=True)


if __name__=='__main__':main()
