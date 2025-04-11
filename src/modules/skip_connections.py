import numpy as np
import torch
from torch import nn


class ResidualConnection(nn.Module):

    def __init__(self, mod: nn.Module, scaling_factor: float = 1/np.sqrt(2)):
        super().__init__()
        self.mod = mod
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor):
        return self.scaling_factor * (x + self.mod(x))


class DenseConnection(nn.Module):

    def __init__(self, mod: nn.Module, feature_dim: int = -1):
        super().__init__()
        self.mod = mod
        self.feature_dim = feature_dim
        self.scaling_factor = 1

    def forward(self, x: torch.Tensor):
        return self.scaling_factor * torch.concat((x, self.mod(x)), dim=self.feature_dim)
