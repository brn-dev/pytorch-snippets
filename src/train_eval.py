import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        val_loader: DataLoader,
        device: torch.device | str,
        disable_tqdm=False,
):
    model.eval()

    losses = []
    with torch.inference_mode():
        for x, y in tqdm(val_loader, disable=disable_tqdm):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())

    return losses


def train(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device | str,
        disable_tqdm=False,
):
    model.train()

    losses = []
    for x, y in tqdm(train_loader, disable=disable_tqdm):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses
