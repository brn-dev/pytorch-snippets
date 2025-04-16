from typing import Any

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
            wandb.log({f'{self.name}/activations': wandb.Histogram(x.detach().cpu().numpy())})

        if self.log_gradients and x.requires_grad:
            x.register_hook(self._grad_hook)

        return x

    def _grad_hook(self, grad: torch.Tensor):
        wandb.log({f'{self.name}/gradients': wandb.Histogram(grad.detach().cpu().numpy())})


def register_wandb_module_hooks(
        module: nn.Module,
        module_name: str,
        log_in_activations: bool = True,
        log_out_activations: bool = True,
        param_names: list = None,
        log_weights: bool = True,
        log_gradients: bool = True
):
    params: dict[str, nn.Parameter]
    if param_names:
        params = {
            name: getattr(module, name)
            for name in param_names
        }
    else:
        params = {}
        log_weights = False
        log_gradients = False

    if log_weights or log_in_activations or log_out_activations:
        def forward_hook(mod, inputs, output):
            log_dict: dict[str, Any] = {}

            if log_in_activations:
                for i, inp in enumerate(inputs):
                    log_dict[f'{module_name}/in_activations/{i}'] = wandb.Histogram(inp.detach().cpu().numpy())
            if log_out_activations:
                log_dict[f'{module_name}/out_activations'] = wandb.Histogram(output.detach().cpu().numpy())

            if log_weights:
                for param_name, param in params.items():
                    log_dict[f"{module_name}/{param_name}_values"] = wandb.Histogram(param.detach().cpu().numpy())

            wandb.log(log_dict)

        module.register_forward_hook(forward_hook)

    if log_gradients:
        for param_name, param in params.items():
            p_name = param_name
            param.register_hook(lambda grad, p_name_=p_name: wandb.log({
                f"{module_name}/{p_name_}_gradients": wandb.Histogram(grad.detach().cpu().numpy())
            }))

    return module
