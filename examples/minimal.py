"""PNG bytes only: this client path needs no torch, numpy, OpenCV or ROS."""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam_ad import Client

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frames',type=Path,required=True,help='Directory containing ordered same-size PNG frames')
parser.add_argument('--text',default='dog')
parser.add_argument('--endpoint',default=os.environ.get('SAM_AD_ENDPOINT','http://127.0.0.1:8850'))
args=parser.parse_args()
frames=sorted(args.frames.glob('*.png'))
if not frames:raise RuntimeError('No PNG frames found: '+str(args.frames))
endpoint=args.endpoint
with Client(endpoint,token=os.environ.get('SAM_AD_API_TOKEN')) as client:
    print(client.wait_ready())
    client.start(args.text)
    for path in frames:
        result=client.process_png(path.read_bytes())
        print(result.data['sequence'],result.data['phase'],result.visible,result.bbox_xyxy)
        if result.mask_png_bytes is not None:
            Path('latest-mask.png').write_bytes(result.mask_png_bytes)
