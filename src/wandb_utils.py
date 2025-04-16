from typing import Any, Literal

import torch
from torch import nn
import wandb


class WnBLogLayerStats(nn.Module):
    def __init__(self, name: str, log_activations=True, log_gradients=True):
        super().__init__()
        self.name = name
        self.log_activations = log_activations
        self.log_gradients = log_gradients

    def forward(self, x: torch.Tensor):
        if self.log_activations:
            wandb.log({f'activations/{self.name}': wandb.Histogram(x.detach().cpu().numpy())})

        if self.log_gradients and x.requires_grad:
            x.register_hook(self._grad_hook)

        return x

    def _grad_hook(self, grad: torch.Tensor):
        wandb.log({f'delta-errors/{self.name}': wandb.Histogram(grad.detach().cpu().numpy())})


def wnb_log_module(
        module: nn.Module,
        module_name: str,
        log_in_activations: bool = True,
        log_out_activations: bool = True,
        param_names: list[str] | Literal['all'] | None = 'all',
        log_weights: bool = True,
        log_gradients: bool = True
):
    params: dict[str, nn.Parameter] = {}

    if param_names:
        params = {
            name: param
            for name, param in module.named_parameters()
            if param_names == 'all' or name in param_names
        }

    if not params:
        log_weights = False
        log_gradients = False

    if log_weights or log_in_activations or log_out_activations:
        def forward_hook(mod, inputs, output):
            log_dict: dict[str, Any] = {}

            if log_in_activations:
                for i, inp in enumerate(inputs):
                    log_dict[f'activations/{module_name}/in/{i}'] = wandb.Histogram(inp.detach().cpu().numpy())
            if log_out_activations:
                log_dict[f'activations/{module_name}/out'] = wandb.Histogram(output.detach().cpu().numpy())

            if log_weights:
                for param_name, param in params.items():
                    log_dict[f"weights/{module_name}/{param_name}"] = wandb.Histogram(param.detach().cpu().numpy())

            wandb.log(log_dict)

        module.register_forward_hook(forward_hook)

    if log_gradients:
        for param_name, param in params.items():
            p_name = param_name
            param.register_hook(lambda grad, p_name_=p_name: wandb.log({
                f"gradients/{module_name}/{p_name_}": wandb.Histogram(grad.detach().cpu().numpy())
            }))

    return module


def wnb_log_modules(
        module: nn.Module,
        module_names: list[str] | dict[str, str] | Literal['all'],
        log_in_activations: bool = True,
        log_out_activations: bool = True,
        param_names: list[str] | Literal['all'] | None = 'all',
        log_weights: bool = True,
        log_gradients: bool = True,
        base_name: str = ''
):
    is_renaming = isinstance(module_names, dict)
    sub_modules: dict[str, nn.Module] = {
        module_names[name] if is_renaming else name: sub_mod
        for name, sub_mod in module.named_modules()
        if module_names == 'all' or name in module_names
    }

    for sub_mod_name, sub_mod in sub_modules.items():
        wnb_log_module(
            module=sub_mod,
            module_name=base_name + sub_mod_name,
            log_in_activations=log_in_activations,
            log_out_activations=log_out_activations,
            param_names=param_names,
            log_weights=log_weights,
            log_gradients=log_gradients,
        )




