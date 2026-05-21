import torch
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Build flavor: {'CUDA' if '+cu' in torch.__version__ else 'CPU-only'}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Sanity check that tensors and basic ops actually work
x = torch.randn(64, 128)
y = torch.randn(128, 32)
z = x @ y
print(f"Tensor matmul test passed. Output shape: {tuple(z.shape)}")