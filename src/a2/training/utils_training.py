import torch


def gpu_available():
    return torch.cuda.is_available()
