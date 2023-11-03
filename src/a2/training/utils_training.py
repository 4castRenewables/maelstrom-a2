import a2.utils.utils

torch = a2.utils.utils._import_torch(__file__)


def gpu_available():
    return torch.cuda.is_available()
