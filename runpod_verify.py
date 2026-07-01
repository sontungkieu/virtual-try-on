import importlib

import torch

print(f"python_cuda_available: {torch.cuda.is_available()}")
print(f"torch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"cuda_runtime: {torch.version.cuda}")

for module_name in ["diffusers", "transformers", "accelerate", "gradio", "fastapi"]:
    module = importlib.import_module(module_name)
    print(f"{module_name}: {getattr(module, '__version__', 'ok')}")
