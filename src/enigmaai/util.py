import torch

def get_device_name() -> str:
    """
    Check if current systam has GPU support.

    Returns:
        str: device name depending of GPU availability
    """
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available(): #and torch.backends.mps.is_built():
        device_name = "mps"
        
    return device_name
    