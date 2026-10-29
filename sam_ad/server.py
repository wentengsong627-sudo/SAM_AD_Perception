"""Portable A/D HTTP service; the packaged GPU runtime is loaded lazily."""
import argparse
import hashlib
import json
import logging
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .contract import Contract, RequestError

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("sam_ad")
PROTOCOL = "flyco-text-sam-v1"
API_REVISION = "20261027-v1"


def source_hash(root=ROOT):
    digest=hashlib.sha256()
    for directory in (root/'sam_ad',root/'runtime_api',root/'runtime_app'):
        for path in sorted(directory.rglob('*.py')):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def create_server(backend, root=ROOT, port=8850, bind="127.0.0.1", token=None, admin_token=None,
                  idle_seconds=180, origins=(), executor=None):
    root=Path(root).resolve()
    if backend not in ('A','D'):raise ValueError('backend must be A or D')
    if bind not in ('127.0.0.1','localhost','::1') and (not token or len(token)<24):
        raise ValueError('A non-loopback bind requires SAM_AD_API_TOKEN with at least 24 characters')
    if executor is None:
        from runtime_api.inference import ModelExecutor
        executor=ModelExecutor(backend,root,root/'runtime_app')
    contract=Contract(idle_seconds)
    fingerprint=source_hash(root)
    run_id=secrets.token_hex(16)

    def health():
        info={'protocol':PROTOCOL,'service_revision':'20261026-r3','api_revision':API_REVISION,
              'api_version':'1.0','backend':backend,'status':'loading','pid':os.getpid(),'run_id':run_id,
              'source_fingerprint':fingerprint,'max_objects':1,'idle_timeout_seconds':idle_seconds,
              'scope':'text-to-2D mask and guarded identity tracking; no depth, planning or flight'}
        if executor.ready.is_set():
            if executor.error is not None:
                info.update(status='error',error=str(executor.error))
            else:
                info.update(executor.host.metadata)
                info.update(status='ready',busy=executor.host.session is not None,
                            process_calls=executor.host.process_calls)
                info['loaded_modules']={name:str(Path(module.__file__).resolve())
                    for name in ('text_sam','abc_sam','sam2','sam3')
                    if (module:=sys.modules.get(name)) is not None and getattr(module,'__file__',None)}
        return info

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        server_version='SAM-AD/1.0'
        disable_nagle_algorithm=True

        def log_message(self,*args):pass

        def respond(self,data,status=200,mime='application/json; charset=utf-8'):
            if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            origin=self.headers.get('Origin')
            if origin in origins:
                self.send_header('Access-Control-Allow-Origin',origin)
                self.send_header('Vary','Origin')
                self.send_header('Access-Control-Allow-Headers','Authorization, Content-Type')
                self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def authorized(self):
            if token and not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.close_connection=True
                self.respond({'error':'Missing or invalid bearer token','error_code':'UNAUTHORIZED'},401)
                return False
            return True

        def do_OPTIONS(self):
            if self.headers.get('Origin') not in origins:
                return self.respond({'error':'Origin is not allowed','error_code':'ORIGIN_NOT_ALLOWED'},403)
            self.respond({},200)

        def do_GET(self):
            if not self.authorized():return
            path=urlsplit(self.path).path
            if path in ('/health','/api/health'):return self.respond(health())
            if path=='/openapi.json':
                return self.respond((root/'openapi.json').read_bytes())
            if path in ('/','/docs'):
                return self.respond((root/'接口说明.html').read_bytes(),mime='text/html; charset=utf-8')
            self.respond({'error':'Not found','error_code':'NOT_FOUND'},404)

        def do_POST(self):
            if not self.authorized():return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16*1024*1024:
                    self.close_connection=True
                    raise RequestError('REQUEST_TOO_LARGE','JSON body must be 1 byte to 16 MiB',413)
                if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
                    self.close_connection=True
                    raise RequestError('INVALID_CONTENT_TYPE','Use application/json',415)
                self.connection.settimeout(65)
                data=json.loads(self.rfile.read(length))
                path=urlsplit(self.path).path
                if path=='/_admin/shutdown':
                    if not admin_token or not isinstance(data,dict) or not secrets.compare_digest(str(data.get('token','')),admin_token):
                        raise RequestError('FORBIDDEN','Administrative token required',403)
                    self.respond({'shutting_down':True})
                    threading.Thread(target=self.server.shutdown,daemon=True).start()
                    return
                if path not in ('/session','/frame','/hint','/reset'):
                    return self.respond({'error':'Not found','error_code':'NOT_FOUND'},404)
                state=health()
                if state['status']!='ready':
                    raise RequestError('MODEL_NOT_READY',state.get('error','Model is loading'),503)
                result=executor.submit(lambda host:contract.dispatch(host,path,data)).result(timeout=60)
                self.respond(result)
            except RequestError as exc:
                self.respond({'error':str(exc),'error_code':exc.code},exc.status)
            except (ValueError,KeyError,TypeError) as exc:
                # The core raises ProtocolError (a ValueError) for session ownership/order.
                message=str(exc)
                conflict=any(word in message.lower() for word in ('session','consecutive','sequence','repeated frame','busy'))
                self.respond({'error':message,'error_code':'SESSION_CONFLICT' if conflict else 'INVALID_REQUEST'},409 if conflict else 422)
            except TimeoutError:
                self.respond({'error':'Inference response timed out; retry the identical sequence and PNG',
                              'error_code':'INFERENCE_TIMEOUT'},504)
            except RuntimeError as exc:
                if 'queue is busy' in str(exc).lower():
                    self.respond({'error':str(exc),'error_code':'QUEUE_BUSY'},503)
                else:
                    LOG.exception('Inference runtime failed')
                    self.respond({'error':str(exc),'error_code':'INTERNAL_ERROR'},500)
            except Exception as exc:
                LOG.exception('Inference request failed')
                self.respond({'error':type(exc).__name__+': '+str(exc),'error_code':'INTERNAL_ERROR'},500)

    server=ThreadingHTTPServer((bind,int(port)),Handler)
    server.daemon_threads=True
    server.executor,server.contract,server.health=executor,contract,health
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['A','D'],required=True)
    parser.add_argument('--port',type=int)
    parser.add_argument('--bind',default='127.0.0.1')
    parser.add_argument('--token-env',default='SAM_AD_API_TOKEN')
    parser.add_argument('--idle-seconds',type=float,default=180)
    parser.add_argument('--allow-origin',action='append',default=[])
    parser.add_argument('--skip-integrity',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    # CUDA server must not inherit an RKNN/ORT board profile from the parent shell.
    for name in ('DINO_FFN_RKNN','SAM_LOW_MEMORY','SAM_DINO_SHORT_EDGE','SAM_DEFORMABLE_LIB','SAM_MEMENC_BACKEND'):
        os.environ.pop(name,None)
    os.environ.update(SAM_ENCODER_BACKEND='torch',SAM_MEMORY_BACKEND='torch',PYTHONUTF8='1')
    if not args.skip_integrity:
        from .doctor import verify_manifest
        verify_manifest(ROOT,args.backend)
    port=args.port or (8850 if args.backend=='A' else 8851)
    server=create_server(args.backend,port=port,bind=args.bind,
        token=os.environ.get(args.token_env),admin_token=os.environ.get('SAM_AD_ADMIN_TOKEN'),
        idle_seconds=args.idle_seconds,origins=tuple(args.allow_origin))
    (ROOT/'runtime').mkdir(exist_ok=True)
    (ROOT/'runtime'/('service-%d.json'%port)).write_text(json.dumps({'pid':os.getpid(),'backend':args.backend,
        'endpoint':'http://'+args.bind+':'+str(port),'run_id':server.health()['run_id']}),encoding='utf-8')
    print('Listening: http://%s:%d/ backend=%s'%(args.bind,port,args.backend),flush=True)
    shutdown=threading.Event()
    def sweep():
        while not shutdown.wait(5):
            if server.executor.ready.is_set() and server.executor.error is None:
                try:server.executor.submit(server.contract.expire)
                except Exception:pass
    thread=threading.Thread(target=sweep,daemon=True)
    thread.start()
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:
        shutdown.set()
        server.server_close()
        server.executor.close()


if __name__=='__main__':main()
