"""Video-file caller; replace cap.read() with a camera/ROS frame in another system."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam_ad import Client


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['A','D'],default='A')
    parser.add_argument('--endpoint')
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--text',default='dog')
    parser.add_argument('--output',type=Path,default=Path('client-output'))
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--display',action='store_true')
    args=parser.parse_args()
    import cv2
    import numpy as np
    endpoint=args.endpoint or 'http://127.0.0.1:'+('8850' if args.backend=='A' else '8851')
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise RuntimeError('Cannot open video: '+str(args.video))
    args.output.mkdir(parents=True,exist_ok=True)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    index=0
    try:
        with Client(endpoint,backend=args.backend,token=os.environ.get('SAM_AD_API_TOKEN')) as client:
            client.wait_ready()
            client.start(args.text)
            with (args.output/'events.jsonl').open('w',encoding='utf-8') as events:
                while args.max_frames is None or index<args.max_frames:
                    ok,bgr=cap.read()
                    if not ok:break
                    # Both schemes see exactly this 640x480 BGR source; the SDK handles RGB conversion.
                    bgr=cv2.resize(bgr,(640,480))
                    result=client.process_bgr(bgr,timestamp_ns=round(index/fps*1_000_000_000))
                    mask=result.mask
                    preview=bgr.copy()
                    if mask is not None:
                        cv2.imwrite(str(args.output/('mask-%06d.png'%index)),mask)
                        visible=mask>0
                        preview[visible]=(preview[visible]*.6+np.array([245,171,50])*.4).astype(np.uint8)
                    # mask=None means no reliable observation. Do not treat a retained ID/state as a visible target.
                    events.write(json.dumps(result.data,ensure_ascii=False)+'\n')
                    if index%10==0:
                        cv2.imwrite(str(args.output/'last_frame.jpg'),preview)
                        print(json.dumps({'frame':index,'phase':result.data['phase'],'visible':result.visible,
                                          'bbox_xyxy':result.bbox_xyxy,'states':result.states},ensure_ascii=False))
                    if args.display:
                        cv2.imshow('SAM A/D | '+args.backend,preview)
                        if cv2.waitKey(1)&255==27:break
                    index+=1
    finally:
        cap.release()
        if args.display:cv2.destroyAllWindows()
    print('Completed %d frames. Session was released.'%index)


if __name__=='__main__':main()
