from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_floating_point() and tensor.dtype in {torch.float16, torch.bfloat16}:
        tensor = tensor.float()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.numpy()

