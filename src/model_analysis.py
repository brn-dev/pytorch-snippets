import numpy as np
import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True):
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)

def check_output_moments(
        model: nn.Module,
        input_shape: tuple[int],
        batch_size: int,
        device: torch.device | str
):
    x = torch.tensor(np.random.normal(size=(batch_size,) + input_shape), dtype=torch.float32).to(device)
    y = model(x)
    y_flat = torch.flatten(y, start_dim=1)

    mean = y_flat.mean(dim=1, keepdim=True)

    variance = torch.var(y_flat - mean, dim=1)
    sigma = variance.sqrt()

    skewness = ((y_flat - mean) ** 3).mean(dim=1) / (sigma ** 3 + 1e-8)
    kurtosis = ((y_flat - mean) ** 4).mean(dim=1) / (sigma ** 4 + 1e-8)

    print(f'mean     = {mean.mean().cpu().item(): .4f}, \n'
          f'variance = {variance.mean().cpu().item(): .4f}, \n'
          f'skewness = {skewness.mean().cpu().item(): .4f}, \n'
          f'kurtosis = {kurtosis.mean().cpu().item(): .4f}')
    print()
