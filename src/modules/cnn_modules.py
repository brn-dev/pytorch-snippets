import torch
from torch import nn



class GlobalAvgPooling2d(nn.Module):
    def __init__(self, keepdims: bool = True):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x: torch.Tensor):
        return x.mean(dim=(-2, -1), keepdim=self.keepdims)


class GlobalMaxPooling2d(nn.Module):
    def __init__(self, keepdims: bool = True):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x: torch.Tensor):
        return x.amax(dim=(-2, -1), keepdim=self.keepdims)


# Depthwise Separable Convolution
# https://arxiv.org/abs/1610.02357
class DSConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int | str = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      groups=in_channels, padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


# https://arxiv.org/abs/1709.01507
class SqueezeAndExcite2d(nn.Module):
    def __init__(self, in_out_channels: int, hidden_channels: int):
        super().__init__()
        self.scaling = nn.Sequential(
            GlobalAvgPooling2d(keepdims=True),
            nn.Conv2d(in_out_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.scaling(x)


