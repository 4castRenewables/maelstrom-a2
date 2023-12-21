import a2.utils.utils

torch = a2.utils.utils._import_torch(__file__)


def cuda_available():
    return torch.cuda.is_available()


def available_cuda_devices():
    return [torch.cuda.device(i) for i in range(torch.cuda.device_count())]


def available_cuda_devices_names():
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
