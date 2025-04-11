import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from src.debug_modules import PrintShape, PrintMoments
from src.skip_connections import ResidualConnection, DenseConnection
from src.training import train


class TestDS(Dataset):

    def __init__(self, num_features: int = 500, size: int = 1000):
        self.num_features = num_features
        self.i = 0
        self.size = size

    def __getitem__(self, idx: int):
        return (
            torch.tensor(np.random.normal(size=self.num_features), dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32)
        )

    def __len__(self):
        return self.size


def main():
    device = torch.device('cuda:0')
    num_features = 10

    train_ds = TestDS(num_features=num_features, size=1000)
    val_ds = TestDS(num_features=num_features, size=1000)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = nn.Linear(num_features, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    train(
        num_epochs=10,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )


if __name__ == "__main__":
    main()
