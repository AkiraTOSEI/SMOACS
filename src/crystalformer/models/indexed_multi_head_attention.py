
import warnings
from typing import List, Optional, Tuple

import math
import torch
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_, normal_
from torch.nn import Parameter, Module
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable

# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        try:
            super().__init__(in_features, out_features, bias=bias,
                            device=device, dtype=dtype)
        except:
            super().__init__(in_features, out_features, bias=bias)


#
# multihead attention
#


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    #assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    #assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    batch: Tensor,
    batch_kv: Tensor,
    edges: Tensor,
    attn_weights: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        edges: index pairs (i,j) to define attentions between q and p,v.
        attn_weights: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(Nt, B, E)` where Nt is the target sequence length, B is batch size,
            and E is embedding dimension.
        - key: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - value: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - edges: :math:`(2, M)` where M is the edge num.
        - attn_weights: `(M, B)` where M in the edge num, B is batch size.
        - Output: attention values have shape :math:`(Nt, B, E)`; attention weights
            have shape :math:`(M, B)` where M in the edge num, B is batch size.
    """
    Nt, B, E = q.shape
    q = q / math.sqrt(E)
    # (M, B, E) x (M, B, E) -> (M, B)
    attn = (q[edges[0]]*k[edges[1]]).sum(dim=-1)

    if attn_weights is not None:
        attn += attn_weights
    
    #flag = torch.are_deterministic_algorithms_enabled()         
    #torch.use_deterministic_algorithms(False)
    bsz = batch.max().item()+1
    q_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
    q_sizes.scatter_add_(0, batch, torch.ones_like(batch))

    if batch_kv is batch:
        k_sizes = q_sizes
    else:
        k_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
        k_sizes.scatter_add_(0, batch_kv, torch.ones_like(batch_kv))
    # This is because self-attention has the same number of queries and keys (sys_size).
    edg_sizes = q_sizes*k_sizes

    q_sizes = q_sizes.tolist()
    k_sizes = k_sizes.tolist()
    edg_sizes = edg_sizes.tolist()
    #torch.use_deterministic_algorithms(flag)

    if True:
        # The scaled_dot operation involves the summations along the key axis
        # whose size varies among batch samples. So we split concatenated data 
        # into a list of batch samples and apply the scaled_dot for each sample.
        # We could do the same without the splitting & looping by using scatter_add,
        # but we rather avoid scatter_add as it breaks reproducibility in backprop.
        attn = torch.split_with_sizes(attn, edg_sizes)
        attn = torch.cat([F.softmax(a.view(qs,ks,-1),dim=1).view(qs*ks,-1) for a,qs,ks in zip(attn,q_sizes,k_sizes)])
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        
        # (M, B, 1) x (M, B, E) -> (q_len, B, E)
        output = attn[...,None]*v[edges[1]]
        output = torch.split_with_sizes(output, edg_sizes)
        output = torch.cat([o.view((qs,ks)+o.shape[1:]).sum(dim=1) for o,qs,ks in zip(output,q_sizes,k_sizes)])
    else:
        # This code was slower (3.65 it/sec vs 3.95 it/sec).
        attn = torch.split_with_sizes(attn, edg_sizes)
        v = torch.split_with_sizes(v, sys_sizes)
        output = []
        for a,v,s in zip(attn,v,sys_sizes):
            a = F.softmax(a.view(s,s,-1), dim=1)
            if dropout_p > 0.0:
                a = F.dropout(a, p=dropout_p)
            # (Nt,Nt,B)x(1,Nt,B,E).sum(dim=1) -> # (Nt,B,E)
            output.append((a[...,None]*v[None]).sum(dim=1))
        output = torch.cat(output)

    return output, attn


def _mha_shape_check(
    query: Tensor, key: Tensor, value: Tensor, edges: Tensor, num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `dist`, and `edges`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    assert query.dim() == 2, f"Expected `query` to be 2-D but got a {query.dim()}-D tensor."
    assert key.dim() == 2, f"Expected `key` to be 2-D but got a {key.dim()}-D tensor."
    assert value.dim() == 2, f"Expected `value` to be 2-D but got a {value.dim()}-D tensor."
    
    assert edges.dim() == 2, f"Expected `edges` to be 2-D but got a {edges.dim()}-D tensor."
    assert edges.shape[0] == 2
    assert edges.dtype == torch.long


def indexed_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    batch_q: Tensor,
    batch_kv: Tensor,
    edges: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    need_weights: bool = True,
    value_activation: Callable[[Tensor], Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        dist: distance matrices of points of lattices.
        lattice_pos_weights: weights for lattice position embeddings.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - dist: :math:`(N, L, S, R)` or `(L, S, R)`, where N is the batch size, S is the source sequence length, 
           N is the batch size, R is the number of neighbors of the lattice.
        - lattice_pos_weights: :math:`(E)`, where E is the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    _mha_shape_check(query, key, value, edges, num_heads)

    # set up shape vars
    tgt_len, embed_dim = query.shape
    src_len, _ = key.shape
    esz = edges.shape[1]
    out_dim = v_proj_weight.shape[0]
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    
    # allow MHA to have different embedding dimensions when separate projection weights are used
    assert key.shape[0] == value.shape[0], \
        f"key's sequence and batch dims {key.shape[0]} do not match value's {value.shape[0]}"
   
    #
    # compute in-projection
    #
    assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
    assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
    assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
    if in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        dq, dk, dv = q_proj_weight.shape[0], k_proj_weight.shape[0], v_proj_weight.shape[0]
        b_q, b_k, b_v = torch.split_with_sizes(in_proj_bias, [dq,dk,dv])
    q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    if value_activation is not None:
        v = value_activation(v)
    #
    # reshape q, k, v for multihead attention
    #
    q = q.contiguous().view(tgt_len, num_heads, head_dim)
    k = k.contiguous().view(k.shape[0], num_heads, head_dim)
    v = v.contiguous().view(v.shape[0], num_heads, out_dim//num_heads)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, batch_q, batch_kv, edges, dropout_p=dropout_p)
    attn_output = attn_output.contiguous().view(tgt_len, out_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(esz, num_heads)
        attn_output_weights = attn_output_weights.mean(dim=1)
        return attn_output, attn_output_weights
    else:
        return attn_output, None

class IndexedMultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, 
                 odim=None, kdim=None, vdim=None, 
                 value_activation: Callable[[Tensor], Tensor] = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(IndexedMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.odim = odim if odim is not None else embed_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((self.odim, self.vdim), **factory_kwargs))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(2 * embed_dim + self.odim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(self.odim, self.odim, bias=bias, **factory_kwargs)
        self.val_act = value_activation

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.out_proj.weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, batch_q:Tensor, batch_kv:Tensor, edges: Tensor,
                need_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """

        attn_output, attn_output_weights = indexed_multi_head_attention_forward(
            query, key, value, batch_q, batch_kv, edges, 
            self.embed_dim, self.num_heads,
            self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
            self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            value_activation=self.val_act,
        )
        return attn_output, attn_output_weights