import torch
from torch import nn

# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import math
from typing import Optional
import einops
import torch
import torch.nn.functional as F
from torch import nn
def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param
def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param

class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, normalized_shape=(self.ndim,), weight=self.weight_proxy, bias=self.bias, eps=self.eps
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
class MultiHeadLayerNorm(LayerNorm):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = input.shape

        gn_in_1 = input.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out

def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: torch.Tensor = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        lower_triangular_matrix (torch.Tensor, optional): (S,S). Defaults to None.
        stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


def recurrent_step_stabilized_simple(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        c_state (torch.Tensor): (B, NH, DH, DH)
        n_state (torch.Tensor): (B, NH, DH, 1)
        m_state (torch.Tensor): (B, NH, 1, 1)
        q (torch.Tensor): (B, NH, 1, DH)
        k (torch.Tensor): (B, NH, 1, DH)
        v (torch.Tensor): (B, NH, 1, DH)
        igate_preact (torch.Tensor): (B, NH, 1, 1)
        fgate_preact (torch.Tensor): (B, NH, 1, 1)

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            (hidden_state [B, NH, DH], (c_state_new [B, NH, DH, DH], n_state_new [B, NH, DH, 1]], m_state_new [B, NH, 1, 1]))
    """
    B, NH, S, DH = q.shape
    # projections
    q, k, v = q.squeeze_(2).unsqueeze(-1), k.squeeze_(2).unsqueeze(-1), v.squeeze_(2).unsqueeze(-1)  # (B, NH, DH, 1)

    # gates
    log_fg_act = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, 1, 1)

    # update rule
    m_state_new = torch.max(log_fg_act + m_state, igate_preact)  # (B, NH, 1, 1)

    fg_act = torch.exp(log_fg_act + m_state - m_state_new)  # (B, NH, 1, 1)
    ig_act = torch.exp(igate_preact - m_state_new)  # (B, NH, 1, 1)

    k_scaled = k / math.sqrt(DH)

    c_state_new = fg_act * c_state + ig_act * (k_scaled @ v.transpose(-1, -2))  # (B, NH, DH, DH)
    n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)

    h_num = q.transpose(-1, -2) @ c_state_new  # (B, NH, 1, DH)

    qn_dotproduct = q.transpose(-1, -2) @ n_state_new  # (B, NH, 1, 1)
    max_val = torch.exp(-m_state_new)  # (B, NH, 1, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + eps
    h = h_num / h_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)

    return h, (c_state_new, n_state_new, m_state_new)


def chunkwise_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,  # B, NH, S, DH
    values: torch.Tensor,  # B, NH, S, DH
    igate_preact: torch.Tensor,  # B, NH, S
    fgate_preact: torch.Tensor,  # B, NH, S
    initial_C: Optional[torch.Tensor] = None,  # B, NH, DH, DH
    initial_n: Optional[torch.Tensor] = None,  # B, NH, DH
    initial_m: Optional[torch.Tensor] = None,  # B, NH, 1
    chunk_size: int = 64,  # optimize this
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    B, NH, S, DH = queries.shape
    NS, CS = S // chunk_size, chunk_size
    _dtype, _device = queries.dtype, queries.device

    # form chunks
    q = queries.view(B, NH, NS, CS, DH) / math.sqrt(DH)
    k = keys.view(B, NH, NS, CS, DH)
    v = values.view(B, NH, NS, CS, DH)

    # forget gates
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact).view(B, NH, NS, CS)
    log_fgates_acc = log_fgates.cumsum(dim=3)
    igate_preact = igate_preact.view(B, NH, NS, CS)

    loggates = (igate_preact - log_fgates_acc)[:, :, :, :, None]
    m_loc, _ = torch.max(
        loggates + log_fgates_acc[:, :, :, -1, None, None], dim=3, keepdim=True
    )
    loggates = loggates + log_fgates_acc[:, :, :, -1, None, None] - m_loc

    kv = k.transpose(-1, -2) @ (v * (loggates).exp())
    ksum = (k * (loggates).exp()).sum(dim=-2)
    C = torch.zeros((B, NH, NS + 1, DH, DH), device=kv.device, dtype=kv.dtype)
    n = torch.zeros((B, NH, NS + 1, DH), device=kv.device, dtype=kv.dtype)
    if initial_C is not None:
        C[:, :, 0] = initial_C
    # print(initial_n.shape)
    if initial_n is not None:
        n[:, :, 0] = initial_n

    m = torch.zeros((B, NH, NS + 1, 1, 1), device=kv.device, dtype=kv.dtype)
    if initial_m is not None:
        # print(initial_m[:, :, None, None].shape)
        # print(m[:, :, 0].shape)
        # m[:, :, 0] = initial_m[:, :, None, None]
        m[:, :, 0] = initial_m

    for i in range(1, NS + 1):
        m[:, :, i] = torch.maximum(
            log_fgates_acc[:, :, i - 1, -1, None, None] + m[:, :, i - 1],
            m_loc[:, :, i - 1],
        )
        C[:, :, i] = (
            C[:, :, i - 1].clone()
            * (
                log_fgates_acc[:, :, i - 1, -1, None, None]
                + m[:, :, i - 1]
                - m[:, :, i]
            ).exp()
            + kv[:, :, i - 1] * (m_loc[:, :, i - 1] - m[:, :, i]).exp()
        )
        n[:, :, i] = (
            n[:, :, i - 1].clone()
            * (
                log_fgates_acc[:, :, i - 1, -1, None]
                + m[:, :, i - 1, 0]
                - m[:, :, i, 0]
            ).exp()
            + ksum[:, :, i - 1] * (m_loc[:, :, i - 1, 0] - m[:, :, i, 0]).exp()
        )

    log_fgates_rep = log_fgates_acc[:, :, :, :, None].repeat(1, 1, 1, 1, CS)
    log_fg_matrix = (
        log_fgates_rep
        - log_fgates_rep.transpose(-1, -2)
        - torch.triu(float("inf") * torch.ones([1, 1, 1, CS, CS]).to(q), diagonal=1)
    )

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact[:, :, :, :, None].transpose(
        -2, -1
    )  # (B, NH, NS, CS, CS)
    D_max, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)

    stab = torch.maximum(D_max, m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None])
    inter_C = (
        q * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()
    ) @ C[:, :, :-1]
    inter_n = (
        q * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()
    ) @ n[:, :, :-1, :, None]

    # D matrix stabilization
    log_D_matrix_stabilized = log_D_matrix - stab  # (B, NH, NS, CS, CS)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, NS, CS, CS)

    # combination matrix C
    qk_matrix = q @ k.transpose(-2, -1)  # (B, NH, NS, CS, CS)
    E_matrix = qk_matrix * D_matrix  # (B, NH, NS, CS, CS)

    normalizer = torch.maximum(
        (E_matrix.sum(dim=-1, keepdim=True) + inter_n).abs(),
        torch.exp(-stab),
    )  # (B, NH, NS, CS, 1)

    E_matrix_normalized = E_matrix / (normalizer + eps)

    # retrieved values
    intra = E_matrix_normalized @ v  # (B, NH, S, DH)
    inter = inter_C / (normalizer + eps)

    if return_last_state:
        return (intra + inter).view((B, NH, S, DH)), (C[:, :, -1], n[:, :, -1], m[:, :, -1])
    else:
        return (intra + inter).view((B, NH, S, DH))

class mLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = chunkwise_simple
        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)

        self.reset_parameters()
    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, S, _ = q.shape  # (B, S, H)
        # assert S == 1, f"mLSTMCell.step only supports sequence length S=1, but got S={S}."

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        _, _, NH, DH = q.shape

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)

        if mlstm_state is None:
            c_state = None
            n_state = None
            m_state = None
        else:
            c_state, n_state, m_state = mlstm_state
            c_state = c_state.to(device=q.device, dtype=q.dtype)
            n_state = n_state.to(device=q.device, dtype=q.dtype)
            m_state = m_state.to(device=q.device, dtype=q.dtype)

        h_state, mlstm_state = self.backend_fn_step(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            initial_C=c_state,
            initial_n=n_state,
            initial_m=m_state,
            chunk_size=16,
            return_last_state=True,
            eps=1e-6
        )  # (B, NH, 1 DH), ((B, NH, DH, DH), (B, NH, DH, 1), (B, NH, 1, 1))

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm, mlstm_state

class mLSTMLayer(nn.Module):
    def __init__(
            self,
            dim=192,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            init_weights="original",
            seqlens=None,
            num_blocks=None,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind
        self.init_weights = init_weights
        self.num_blocks = num_blocks

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )

        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        self.conv_act_fn = nn.SiLU()

        self.mlstm_cell = mLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )
        self.ogate_act_fn = nn.SiLU()

        self.learnable_skip = nn.Parameter(torch.ones(inner_dim, requires_grad=True))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        # self.dropout = nn.Dropout(self.config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        if self.init_weights == "original":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        elif self.init_weights == "original-fixed":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=self.num_blocks)
        else:
            raise NotImplementedError
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        B, S, _ = x.shape

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state, mlstm_state = self.mlstm_cell(q=q, k=k, v=v, mlstm_state=mlstm_state)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # down-projection
        y = self.proj_down(h_state)
        return y, mlstm_state
    
if __name__ == '__main__':
    x = torch.rand(64, 16, 192)
    m = mLSTMLayer()
    y, state = m(x, None)
    print(y.shape)
    print(state["mlstm_state"][0].shape)
    pass
