"""Wire/SDK contract tests with an explicit fake pipeline, not GPU quality tests."""
import base64
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT if (ROOT/'sam_ad').is_dir() else ROOT/'outputs/handoff/SAM_AD_Module_20261027'
sys.path.insert(0,str(BUNDLE))
from runtime_api.inference import InferenceHost
from sam_ad import APIError, Client, TransportError
from sam_ad.server import create_server


class Pipeline:
    def __init__(self):
        self.next_id=1; self.tracks={}; self.memory_updates=self.sam_encoder_calls=self.detection_count=0
        self.tracker=SimpleNamespace(reset=lambda:None); self.last_detection_frame=None

    def state(self,track,state,*_):track['state']=state

    def process(self,rgb,sequence,frames,command):
        identity=next(iter(self.tracks),self.next_id)
        self.tracks[identity]={'id':identity,'state':'VERIFYING' if sequence==0 else 'TRACKING','verified':sequence>0}
        masks={}
        if sequence>0:
            mask=np.zeros(rgb.shape[:2],bool); mask[2:6,3:9]=True
            masks={identity:mask}; self.memory_updates+=1
        self.sam_encoder_calls+=1
        return {'masks':masks,'phase':'tracking' if masks else 'searching','objects':list(self.tracks.values())}


def parse(text):
    return SimpleNamespace(quantity='all' if text.startswith('all ') else 'one',query=text+'.',action='stop' if text=='stop' else 'track')


class Executor:
    def __init__(self,backend='A'):
        tokenizer=SimpleNamespace(encode=lambda text:text.split())
        engine=SimpleNamespace(calls=0,identity_commits=0,model=SimpleNamespace(backbone=SimpleNamespace(language_backbone=SimpleNamespace(tokenizer=tokenizer))))
        self.host=InferenceHost(backend,Pipeline,parse,engine,{'test_pipeline':True})
        self.error=None; self.ready=threading.Event(); self.ready.set(); self.lock=threading.Lock()
    def submit(self,fn):
        future=Future()
        with self.lock:
            try:future.set_result(fn(self.host))
            except BaseException as exc:future.set_exception(exc)
        return future
    def close(self):pass


class WireTests(unittest.TestCase):
    def setUp(self):
        self.executor=Executor()
        self.server=create_server('A',BUNDLE,port=0,executor=self.executor,origins=('http://localhost:3000',))
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True); self.thread.start()
        self.endpoint='http://127.0.0.1:'+str(self.server.server_address[1])
        self.client=Client(self.endpoint,backend='A'); self.client.start('dog')
        self.rgb=np.zeros((16,20,3),np.uint8); self.rgb[:]=[230,70,10]

    def tearDown(self):
        self.client.close(); self.server.shutdown(); self.server.server_close(); self.thread.join(2)

    def wire(self,path,payload=None,method=None,headers=None):
        body=None if payload is None else json.dumps(payload).encode()
        request=urllib.request.Request(self.endpoint+path,body,headers={'Content-Type':'application/json',**(headers or {})},method=method)
        try:
            with urllib.request.urlopen(request,timeout=10) as r:return r.status,json.loads(r.read())
        except urllib.error.HTTPError as e:return e.code,json.loads(e.read())

    def payload(self,sequence=0,timestamp=0):
        _,png=cv2.imencode('.png',cv2.cvtColor(self.rgb,cv2.COLOR_RGB2BGR))
        return {'session':self.client.session,'sequence':sequence,'timestamp_ns':timestamp,'image_png':base64.b64encode(png).decode()}

    def test_rgb_bgr_mask_and_box(self):
        first=self.client.process_rgb(self.rgb,timestamp_ns=0)
        self.assertFalse(first.visible); self.assertIsNone(first.mask); self.assertEqual(first.states,{1:'VERIFYING'})
        result=self.client.process_bgr(self.rgb[:,:,::-1].copy(),timestamp_ns=10)
        self.assertTrue(result.visible); self.assertEqual(result.bbox_xyxy,[3,2,9,6]); self.assertEqual(np.count_nonzero(result.mask),24)
        self.assertEqual(result.data['input_rgb_sha256'],hashlib.sha256(self.rgb.tobytes()).hexdigest())

    def test_duplicate_and_metadata_conflict(self):
        payload=self.payload()
        status,result=self.wire('/frame',payload); self.assertEqual(status,200)
        status,again=self.wire('/frame',payload); self.assertEqual(result,again)
        self.assertEqual(self.executor.host.process_calls,1)
        payload['timestamp_ns']=1
        self.assertEqual(self.wire('/frame',payload)[0],409)
        self.assertEqual(self.executor.host.process_calls,1)

    def test_order_dimensions_bad_png(self):
        self.assertEqual(self.wire('/frame',self.payload(1))[1]['error_code'],'OUT_OF_ORDER')
        self.client.process_rgb(self.rgb)
        with self.assertRaises(APIError) as error:self.client.process_rgb(np.zeros((8,8,3),np.uint8))
        self.assertEqual(error.exception.code,'IMAGE_SIZE_CHANGED')
        payload=self.payload(1); payload['image_png']='bad'
        self.assertEqual(self.wire('/frame',payload)[1]['error_code'],'INVALID_PNG')
        self.assertEqual(self.executor.host.process_calls,1)

    def test_session_ownership_all_and_expiry(self):
        with Client(self.endpoint) as other:
            with self.assertRaises(APIError):other.start('dog')
        with self.assertRaises(APIError) as error:self.client.start('all dogs')
        self.assertEqual(error.exception.code,'SINGLE_TARGET_ONLY')
        self.client.process_rgb(self.rgb)
        old=self.client.session
        self.server.contract.touched-=200
        self.client.start('dog')
        self.assertNotEqual(old,self.client.session)
        self.assertEqual(self.client.sequence,-1)

    def test_hint_invalidates_cached_observation(self):
        self.client.process_rgb(self.rgb)
        result=self.client.process_rgb(self.rgb,timestamp_ns=8)
        self.assertTrue(result.visible)
        hint=self.client.hint(); self.assertEqual(hint['reference_ids'],[1])
        _,cached=self.wire('/frame',self.payload(1,8))
        self.assertIsNone(cached['mask_png']); self.assertFalse(cached['visible']); self.assertEqual(cached['ids'],[])
        self.assertEqual(cached['states'],[[1,'SUSPECT']])

    def test_lost_response_retry_runs_once(self):
        original=self.client._request
        def dropped(path,payload=None,retry=False):
            result=original(path,payload,retry)
            if path=='/frame':raise TransportError('simulated response lost after server commit')
            return result
        self.client._request=dropped
        with self.assertRaises(TransportError):self.client.process_rgb(self.rgb)
        self.assertEqual(self.client.sequence,-1); self.assertEqual(self.executor.host.process_calls,1)
        with self.assertRaises(RuntimeError):self.client.process_rgb(np.full_like(self.rgb,1))
        self.client._request=original
        self.client.retry_pending()
        self.assertEqual(self.client.sequence,0); self.assertEqual(self.executor.host.process_calls,1)

    def test_schema_docs_cors_and_admin(self):
        self.assertEqual(self.wire('/openapi.json')[1]['openapi'],'3.1.0')
        with urllib.request.urlopen(self.endpoint+'/docs') as r:self.assertIn('A / D'.encode(),r.read())
        self.assertEqual(self.wire('/frame',method='OPTIONS',headers={'Origin':'http://localhost:3000'})[0],200)
        self.assertEqual(self.wire('/frame',method='OPTIONS',headers={'Origin':'http://other'})[0],403)
        self.assertEqual(self.wire('/_admin/shutdown',{'token':'no'})[0],403)

    def test_D_long_text_and_no_CPU_fallback_claim(self):
        self.executor.host.backend='D'
        with self.assertRaises(APIError) as error:self.client.start(' '.join(['dog']*15))
        self.assertEqual(error.exception.code,'TEXT_TOO_LONG')
        self.assertTrue(self.server.health()['test_pipeline'])

    def test_auth_and_bind_guard(self):
        with self.assertRaises(ValueError):create_server('A',BUNDLE,bind='0.0.0.0',executor=self.executor)
        server=create_server('A',BUNDLE,port=0,token='a'*24,executor=self.executor)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True); thread.start()
        try:
            endpoint='http://127.0.0.1:'+str(server.server_address[1])
            with Client(endpoint) as c:
                with self.assertRaises(APIError) as error:c.health()
                self.assertEqual(error.exception.status,401)
            c=Client(endpoint,token='a'*24); self.assertEqual(c.health()['status'],'ready'); c.close()
        finally:server.shutdown(); server.server_close(); thread.join(2)


if __name__=='__main__':unittest.main(verbosity=2)
