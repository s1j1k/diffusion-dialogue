import torch

# See scripts/train.py: denormal floats make CPU matmuls very slow
torch.set_flush_denormal(True)
