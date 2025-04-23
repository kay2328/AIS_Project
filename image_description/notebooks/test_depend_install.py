try:
    import torch
    import torchvision
    import transformers
    import accelerate
    import peft
    import tokenizers
    import gradio
    import pandas
    import scipy
    import einops
    import sentencepiece
    import tiktoken
    import google.protobuf
    import uvicorn
    import pydantic
    import fastapi
    import sse_starlette
    import matplotlib
    import fire
    import packaging
    import yaml
    import numpy
    import av
    import librosa
    import tyro
    print("All dependencies imported successfully!")
    # checking for GPU utilization
    print("CUDA available:", torch.cuda.is_available())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

except Exception as e:
    print("Error importing dependencies:", e)