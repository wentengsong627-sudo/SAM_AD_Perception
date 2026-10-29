"""Offline bundle integrity and CUDA checks before handing a service to a caller."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify_manifest(root=ROOT,backend=None):
    root=Path(root).resolve()
    document=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    checked=0
    for item in document['files']:
        if backend and item.get('backend') not in (None,backend):continue
        path=(root/item['path']).resolve()
        path.relative_to(root)
        if not path.is_file() or path.stat().st_size!=item['bytes']:
            raise RuntimeError('Missing or wrong-size bundle file: '+item['path'])
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=item['sha256']:
            raise RuntimeError('SHA256 mismatch: '+item['path'])
        checked+=1
    return {'verified_files':checked,'backend':backend or 'A+D'}


def check(backend=None,integrity=True):
    import torch
    import torchvision
    report={'python':sys.version,'platform':platform.platform(),'CUDA_available':torch.cuda.is_available(),
            'torch':torch.__version__,'torchvision':torchvision.__version__,'torch_CUDA':torch.version.cuda,
            'scope':'host/dependency/integrity checks; not a substitute for a real inference smoke test'}
    if not report['CUDA_available']:raise RuntimeError('CUDA is unavailable; install the NVIDIA driver and the CUDA PyTorch wheel')
    report.update(device=torch.cuda.get_device_name(),VRAM_MiB=round(torch.cuda.get_device_properties(0).total_memory/1048576),
                  capability=torch.cuda.get_device_capability(),wheel_architectures=torch.cuda.get_arch_list())
    test=torch.ones((8,8),device='cuda')
    assert float((test@test).sum())==512
    torchvision.ops.nms(torch.tensor([[0,0,2,2]],device='cuda',dtype=torch.float32),torch.ones(1,device='cuda'),.5)
    report['CUDA_tensor_and_torchvision_nms_passed']=True
    if integrity:report['integrity']=verify_manifest(ROOT,backend)
    report['packages']={name:importlib.metadata.version(name) for name in
        ('numpy','transformers','hydra-core','omegaconf','timm','einops','opencv-python')}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['A','D'])
    parser.add_argument('--skip-integrity',action='store_true')
    args=parser.parse_args()
    print(json.dumps(check(args.backend,not args.skip_integrity),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
