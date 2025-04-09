import numpy as np
import torch
from torch import nn

from src.debug_modules import PrintShape, PrintMoments
from src.skip_connections import ResidualConnection, DenseConnection

if __name__ == "__main__":
    # Create a batched tensor where batch dimension is 3 and each sample has 100 activations.
    x = torch.tensor(np.random.normal(size=(30, 500)), dtype=torch.float32)

    lin = nn.Linear(500, 500)
    nn.init.xavier_normal_(lin.weight)
    nn.init.zeros_(lin.bias)

    PrintMoments()(x)

    nn.Sequential(
        ResidualConnection(lin),
        PrintMoments()
    )(x)

    nn.Sequential(
        DenseConnection(lin),
        PrintMoments()
    )(x)
