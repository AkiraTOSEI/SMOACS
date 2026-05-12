import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from . import cuda_funcs
from . import global_config as config
from .indexed_multi_head_attention import IndexedMultiheadAttention


def max_pool(x, batch, sizes):
    x = torch.split_with_sizes(x, sizes.tolist(), 0)
    x = torch.stack([torch.max(x, dim=0)[0] for x in x])
    return x


def avr_pool(x, batch, sizes):
    if config.REPRODUCIBLITY_STATE >= 1 and cuda_funcs.CUPY_AVAILABLE:
        x = cuda_funcs.IrregularMeanCUDA.apply(x, batch, sizes)
    else:
        x = torch.split_with_sizes(x, sizes.tolist(), 0)
        x = torch.stack([torch.mean(x, dim=0) for x in x])
    return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PoolingByMultiheadAttention(nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        dim_in: int,
        dim_out,
        nhead: int = 8,
        num_seeds: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        no_layer_norm: bool = False,
        value_activation: Callable[[Tensor], Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        super(PoolingByMultiheadAttention, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        factory_kwargs = {}
        self.no_ff_layer = dim_feedforward <= 0
        self.nhead = nhead

        assert num_seeds == 1, "Currently only num_seeds=1 is supported."
        self.seeds = nn.Parameter(torch.Tensor(num_seeds, dim_in))

        self.self_attn = IndexedMultiheadAttention(
            dim_in,
            nhead,
            odim=dim_out,
            dropout=dropout,
            value_activation=value_activation,
            **factory_kwargs
        )
        # Implementation of Feedforward model
        if not self.no_ff_layer:
            self.linear1 = Linear(dim_out, dim_feedforward, **factory_kwargs)
            self.dropout = Dropout(dropout) if dropout > 0 else lambda x: x
            self.linear2 = Linear(dim_feedforward, dim_out, **factory_kwargs)
            self.norm2 = (
                LayerNorm(dim_out, eps=layer_norm_eps, **factory_kwargs)
                if not no_layer_norm
                else (lambda x: x)
            )
            self.dropout2 = Dropout(dropout) if dropout > 0 else lambda x: x

        self.norm_first = norm_first
        self.norm1 = (
            LayerNorm(dim_out, eps=layer_norm_eps, **factory_kwargs)
            if not no_layer_norm
            else (lambda x: x)
        )
        self.dropout1 = Dropout(dropout) if dropout > 0 else lambda x: x

        # Legacy string support for activation function.
        self.activation = (
            _get_activation_fn(activation)
            if isinstance(activation, str)
            else activation
        )
        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: how to init seeds?
        nn.init.xavier_uniform_(self.seeds)
        dim_model = self.seeds.shape[1]
        dim_head = dim_model // self.nhead
        nn.init.normal_(self.seeds, 0, dim_model**-0.5)

        if not self.no_ff_layer:
            xavier_uniform_(self.linear1.weight)
            xavier_uniform_(self.linear2.weight)
            if self.linear1.bias is not None:
                constant_(self.linear1.bias, 0)
            if self.linear2.bias is not None:
                constant_(self.linear2.bias, 0)

    def fixup_initialization(self, num_layers):
        temp_state_dic = {}
        en_layers = num_layers

        # TODO: how to init seeds?
        for name, param in self.named_parameters():
            if name in [
                "linear1.weight",
                "linear2.weight",
                "self_attn.out_proj.weight",
                "self_attn.v_proj_weight",
            ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (-1.0 / 4.0)) * param

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(PoolingByMultiheadAttention, self).__setstate__(state)

    def forward(self, src: Tensor, batch: Tensor, batch_size: int) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = self._at_block(self.norm1(x), batch, batch_size)
            if not self.no_ff_layer:
                x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(self._at_block(x, batch, batch_size))
            if not self.no_ff_layer:
                x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _at_block(self, x: Tensor, batch: Tensor, batch_size: int) -> Tensor:
        s = self.seeds.repeat(batch_size, 1)
        batch_q = torch.arange(batch_size, dtype=batch.dtype, device=batch.device)
        edges = torch.stack(
            [
                batch,
                torch.arange(x.shape[0], dtype=batch.dtype, device=batch.device),
            ]
        )

        x = self.self_attn(s, x, x, batch_q, batch, edges, need_weights=False)[0]
        x = self.dropout1(x)
        if s.shape[1] == x.shape[1]:
            return s + x
        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
