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

    pbar = tqdm(val_loader, disable=disable_tqdm, delay=0.01)

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

            pbar.set_description(f'{tqdm_header} - batch_loss = {loss:7.4f}', refresh=False)

    return losses


def train_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        device: torch.device | str | int,
        clip_grad_norm: float | None = None,
        tqdm_header: str | None = None,
        disable_tqdm=False,
):
    device: torch.device = torch.device(device)
    model.train()

    pbar = tqdm(train_loader, disable=disable_tqdm, delay=0.1)
    if not tqdm_header:
        tqdm_header = 'Train'
    pbar.set_description(tqdm_header, refresh=False)

    losses: list[float] = []
    grad_norms: list[float] = []
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        info = f'batch_loss = {loss:7.4f}'

        if clip_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm).item()
            grad_norms.append(grad_norm)
            info += f', {grad_norm = :7.4f} (clip={clip_grad_norm:.2f})'

        optimizer.step()

        pbar.set_description(f'{tqdm_header} - {info}', refresh=False)

    return losses, grad_norms if clip_grad_norm else None


def train(
        num_epochs: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device | str | int,
        clip_grad_norm: float | None = None,
        scheduler: optim.lr_scheduler.LRScheduler = None,
):
    all_train_losses: list[list[float]] = []
    all_val_losses: list[list[float]] = []

    run_id = datetime.now().strftime('%Y%m%d-%H%S')
    run_path = Path(f'./models/{run_id}')
    run_path.mkdir(parents=True)

    print(f'\nRun {run_id}: Training model with '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters '
          f'for {num_epochs} epochs with learning rate {optimizer.param_groups[0]["lr"]:.2e} '
          f'on device {device} \n', flush=True)

    best_val_loss: float = 1e6

    for epoch in range(1, num_epochs + 1):
        epoch_train_losses, epoch_grad_norms = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            clip_grad_norm=clip_grad_norm,
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

        epoch_val_loss = np.mean(epoch_val_losses)

        info = (f'\nEpoch {epoch:>4}:    '
                f'train_loss = {np.mean(epoch_train_losses):7.4f} ± {np.std(epoch_train_losses):7.4f},     '
                f'val_loss = {epoch_val_loss:7.4f} ± {np.std(epoch_val_losses):7.4f}')
        if clip_grad_norm:
            info += (f',     grad_norm = {np.mean(epoch_grad_norms):7.4f} ± {np.std(epoch_grad_norms):7.4f}  '
                     f'(max={np.max(epoch_grad_norms):.4f})')
        print(info)

        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss

            model_path = run_path.joinpath('best.state-dict.pth')
            torch.save(model.state_dict(),  model_path)

            print(f'Saved best model at {model_path} with loss = {best_val_loss:7.4f}')

        if scheduler is not None:
            scheduler.step()

        print(flush=True)

    return all_train_losses, all_val_losses
