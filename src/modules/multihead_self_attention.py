from typing import Optional

import torch
from torch import nn

class MultiheadSelfAttention(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: int = None,
            vdim: int = None,
            batch_first: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim,
                                         batch_first, device, dtype)

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> (torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]):
        attention_out, attention_out_weights = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if need_weights:
            return attention_out, attention_out_weights
        return attention_out
