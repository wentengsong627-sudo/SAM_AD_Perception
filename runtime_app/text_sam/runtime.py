"""Explicit CPU/CUDA selection and honest host-specific memory reporting."""
import os,platform
from pathlib import Path
import torch

def device():
    requested=os.environ.get('SAM_DEVICE','cuda').strip().lower()
    if requested=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested not in {'cpu','cuda'}:raise ValueError('SAM_DEVICE must be cpu, cuda or auto')
    return requested

def require_device():
    selected=device()
    if selected=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Explicitly set SAM_DEVICE=cpu for CPU inference.')
    return selected

def empty_cache():
    if device()=='cuda' and torch.cuda.is_available():torch.cuda.empty_cache()

def reset_peak_memory_stats():
    if device()=='cuda':torch.cuda.reset_peak_memory_stats()

def gpu_peak_mb():
    return torch.cuda.max_memory_allocated()/1048576 if device()=='cuda' else 0.0

def process_peak_rss_mb():
    # Linux ru_maxrss is KiB and is a process-lifetime high-water mark.
    if platform.system()=='Linux':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    try:
        import psutil
        info=psutil.Process().memory_info()
        return getattr(info,'peak_wset',info.rss)/1048576
    except ImportError:return None

npu_active = False
memory_npu_active = False

def image_sizes():
    if os.environ.get('SAM_ENCODER_BACKEND')=='rknn':
        return [512]
    return [512,768,1024]

def description():
    return {'device':device(),'architecture':platform.machine(),'npu_enabled':npu_active,'sam2_encoder_backend':os.environ.get('SAM_ENCODER_BACKEND','torch'), 'sam2_memory_backend':os.environ.get('SAM_MEMORY_BACKEND','torch'), 'memory_npu_active':memory_npu_active,
            'supported_image_sizes':image_sizes(),'memory_metric':'process_lifetime_peak_rss' if device()=='cpu' else 'cuda_job_peak_allocated',
            'profile':'rk3588-a-optimized' if os.environ.get('SAM_MEMORY_BACKEND')=='ort' else 'rk3588-a-hybrid-v2' if os.environ.get('SAM_MEMORY_BACKEND')=='rknn' else 'rk3588-a-hybrid' if os.environ.get('SAM_ENCODER_BACKEND')=='rknn' else 'rk3588-a-cpu' if device()=='cpu' else 'a-cuda'}
