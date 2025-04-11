from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        val_loader: DataLoader,
        device: torch.device | str,
        tqdm_header: str | None = None,
        disable_tqdm=False,
):
    model.eval()

    pbar = tqdm(val_loader, disable=disable_tqdm)

    if not tqdm_header:
        tqdm_header = 'Train'
    pbar.set_description(tqdm_header, refresh=False)

    losses: list[float] = []
    with torch.inference_mode():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())

            pbar.set_description(f'{tqdm_header} - batch_loss = {loss:8.4f}')

    return losses


def train_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        device: torch.device | str | int,
        tqdm_header: str | None = None,
        disable_tqdm=False,
):
    device: torch.device = torch.device(device)
    model.train()

    pbar = tqdm(train_loader, disable=disable_tqdm)

    if not tqdm_header:
        tqdm_header = 'Train'
    pbar.set_description(tqdm_header, refresh=False)

    losses: list[float] = []
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f'{tqdm_header} - batch_loss = {loss:8.4f}')

    return losses


def train(
        num_epochs: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device | str | int,
        scheduler: optim.lr_scheduler.LRScheduler = None,
):

    all_train_losses: list[list[float]] = []
    all_val_losses: list[list[float]] = []

    run_id = datetime.now().strftime("%Y%m%d-%H%S")
    run_path = Path(f'./models/{run_id}')
    run_path.mkdir(parents=True)

    best_val_loss: float = 1e6

    for epoch in range(1, num_epochs + 1):
        epoch_train_losses = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            tqdm_header=f'Epoch {epoch:>3}/{num_epochs:>3} - Train'
        )
        epoch_val_losses = evaluate(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            tqdm_header=f'Epoch {epoch:>3}/{num_epochs:>3} - Eval'
        )
        all_train_losses.append(epoch_train_losses)
        all_val_losses.append(epoch_val_losses)

        print(f"\nEpoch {epoch:>4}:    "
              f"train_loss = {np.mean(epoch_train_losses):8.4f},     "
              f"val_loss = {np.mean(epoch_val_losses):8.4f}")

        epoch_val_loss = np.mean(epoch_val_losses)
        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss

            model_path = run_path.joinpath('best.state-dict.pth')
            torch.save(model.state_dict(),  model_path)

            print(f'Saved best model at {model_path} with loss = {best_val_loss:7.4f}')

        if scheduler is not None:
            scheduler.step()

        print(flush=True)

    return all_train_losses, all_val_losses
