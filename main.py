import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from src.model_analysis import check_output_moments
from src.modules.debug_modules import PrintMoments
from src.modules.skip_connections import ResidualConnection
from src.training import train
from src.wandb_utils import wnb_log_module, WnBLogLayerStats, wnb_log_modules
from src.weight_init import init_weights_


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
    # wandb.init(project='snippets-test', )

    device = torch.device('cuda:0')
    in_features = 100
    hidden_features = in_features
    out_features = 1

    train_ds = TestDS(num_features=in_features, size=1000)
    val_ds = TestDS(num_features=in_features, size=1000)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        ResidualConnection(
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_features, out_features),
            ),
            nn.Sequential(
                nn.Linear(hidden_features, out_features)
            )
        ),
        WnBLogLayerStats('out')
    ).to(device)
    wnb_log_modules(model, {'0': 'in_layer', '1': 'res'})
    init_weights_(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    # check_output_moments(model, (in_features,), 64, device)

    # x = torch.tensor(np.random.normal(size=(32, in_features)), dtype=torch.float32).to(device)
    # print(torch.var(x))
    # print(torch.var(model(x)))

    train(
        num_epochs=50,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        clip_grad_norm=50,
        wandb_init_cfg={
            'project': 'snippets-test',
        }
    )


if __name__ == "__main__":
    main()
