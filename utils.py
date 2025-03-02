import torch
from config import DEVICE, USE_FLOAT16

def to_device_and_dtype(tensor, device=DEVICE):
    tensor = tensor.contiguous().to(device)
    if USE_FLOAT16 and device != "cpu":
        tensor = tensor.to(torch.float16)
    return tensor.contiguous()

def clear_cache():
    if DEVICE == "mps" and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    elif DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()