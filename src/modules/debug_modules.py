import torch
from torch import nn


class PrintShape(nn.Module):
    def __init__(self, name: str = ''):
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor):
        if self.name:
            print(f'{self.name}: ')
        print(x.shape)
        print()
        return x

class PrintMoments(nn.Module):
    def __init__(self, name: str = ''):
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor):
        x_flat = x.flatten(1)

        mean = x_flat.mean(dim=1, keepdim=True)

        variance = torch.var(x_flat - mean, dim=1)
        sigma = variance.sqrt()

        skewness = ((x_flat - mean) ** 3).mean(dim=1) / (sigma ** 3 + 1e-8)
        kurtosis = ((x_flat - mean) ** 4).mean(dim=1) / (sigma ** 4 + 1e-8)

        if self.name:
            print(f'{self.name}: ')
        print(f'mean     = {mean.mean().cpu().item(): .4f}, \n'
              f'variance = {variance.mean().cpu().item(): .4f}, \n'
              f'skewness = {skewness.mean().cpu().item(): .4f}, \n'
              f'kurtosis = {kurtosis.mean().cpu().item(): .4f}')
        print()

        return x


