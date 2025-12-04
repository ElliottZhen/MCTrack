import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

class MambaBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank, d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor):
        super().__init__()
        #  projects block input from D to 2*ED (two branches)
        self.dt_scale = dt_scale
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)

        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=d_conv, bias=conv_bias,
                                groups=self.d_inner,
                                padding=(d_conv - 1)//2)

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.state_fusion = StateFusion(self.d_inner)

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, h=None):
        #  x : (B,L, D)
        # h : (B,L, ED, N)
        #  y : (B, L, D)
        xz = self.in_proj(x)  # (B, L,2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B,L, ED), (B,L, ED)
        x_cache = x.permute(0,2,1)#(B, ED,L)

        #  x branch
        x = self.conv1d(x_cache).permute(0,2,1) #  (B,L , ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        #y->B,L,ED;h->B,L,ED,N

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        return output, h

    def ssm_step(self, x, h=None):
        #  x : (B, L, ED)
        #  h : (B, L, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state],
                                  dim=-1)  #  (B, L,dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B,L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B,L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L,ED, N)

        if h is None:
            h = torch.zeros(x.size(0), x.size(1), self.d_inner, self.d_state, device=deltaA.device)  #  (B, L, ED, N)

        h = deltaA * h + BX  #  (B, L, ED, N)

        b_h, l_h, d_h, hw = h.shape

        h = rearrange(h, "b l d (h w) -> (b l) d h w", h=int(math.sqrt(hw)), w=int(math.sqrt(hw)))
        h = self.state_fusion(h)
        h = rearrange(h, "(b l) d h w -> b l d (h w)", b=b_h, l=l_h)

        y = (h @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x#B,L,ED

        #  todo : pq h.squeeze(1) ??
        return y, h


class Injector(nn.Module):
    def __init__(self, d_model, n_heads=8,norm_layer=partial(nn.LayerNorm, eps=1e-6),  dropout=0.1,
                 init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,dropout=dropout)
        self.gamma = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

    def forward(self, query,feat):
            #query:l,b,d;feat:l,b,d
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query),
                             self.feat_norm(feat),self.feat_norm(feat))[0]
            return query + self.gamma * attn
        query = _inner_forward(query, feat)
        return query

class InteractionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.injector = Injector(d_model=d_model)

    def forward(self,x,xs):

        x = self.injector(x.permute(1,0,2),xs.permute(1,0,2)).permute(1,0,2)
        return x

class Focus_Neck(nn.Module):
    def __init__(self, in_channel=512,d_model=512,d_inner=1024,bias=False,n_layers=6,dt_rank=32,d_state=16,d_conv=3,dt_min=0.001,
                 dt_max=0.1,dt_init='random',dt_scale=1.0,conv_bias=True,dt_init_floor=0.0001):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.bias = bias
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.dt_scale = dt_scale
        self.num_channels = self.d_model
        self.n_layers = n_layers
        # self.layers = nn.ModuleList(
        #     [ResidualBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
        #      for _ in range(n_layers)])
        self.interactions = nn.ModuleList(
            [InteractionBlock(d_model=d_model)
            for _ in range(n_layers)
        ])

    def forward(self, x, xs):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        for i in range(self.n_layers):
            # xs, h = self.layers[i](xs, h)
            x = self.interactions[i](x, xs)

        return x
    
class ResidualBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor):
        super().__init__()

        self.mixer = MambaBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
        self.norm = RMSNorm(d_model)

    def forward(self, x, h):
        #  x : (B, L, D)
        # h : (B, L, ED, N)
        #  output : (B,L, D)

        x = self.norm(x)
        output, h = self.mixer(x, h)
        output = output + x
        return output, h


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()
        
        #self.dim = dim
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)

        self.alpha = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, h):
        h1 = self.conv1(h)
        h2 = self.conv2(h)
        out = self.alpha[0]*h1 + self.alpha[1]*h2
        return out

# class PoolingAttention(nn.Module):
#     def __init__(self, dim=192, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
#                  pool_ratios=[1, 4, 9, 16]):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
#         self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.pool_ratios = pool_ratios
#         self.pools = nn.ModuleList()
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         B, N, C = x.shape

#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         pools = []
#         x_ = x.permute(0, 2, 1)
#         x_1, x_2, x_3 = torch.chunk(x_, chunks=3, dim=2)
#         pooling_banks = [x_1, x_2, x_3]
#         for i in range(len(pooling_banks)):
#             for pool_ratio in self.pool_ratios:
#                 pool = F.adaptive_avg_pool1d(pooling_banks[i], pool_ratio)
#                 pools.append(pool)

#         pools = torch.cat(pools, dim=2)
#         pools = self.norm(pools.permute(0, 2, 1))

#         kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v)
#         x = x.transpose(1, 2).contiguous().reshape(B, N, C)

#         x = self.proj(x)

#         return x

    # def scale_aware(self, x, win_num, pool_ratios):

    #     B, N, C = x.shape
    #     # Side = int(math.sqrt(N // win_num))
    #     pools = []
    #     x_ = x.permute(0, 2, 1)
    #     x_1, x_2, x_3 = torch.chunk(x_, chunks=3, dim=2)
    #     pooling_banks = [x_1, x_2, x_3]
    #     for i in range(len(pooling_banks)):
    #         for pool_ratio in pool_ratios:
    #             # pool = F.adaptive_avg_pool2d(pooling_banks[i], (Side // pool_ratio, Side // pool_ratio))
    #             pool = F.adaptive_avg_pool1d(pooling_banks[i], pool_ratio)
    #             pools.append(pool)
    #     pools = torch.cat(pools, dim=2)

    #     return pools.permute(0, 2, 1)

# class Injector(nn.Module):
#     def __init__(self, d_model, n_heads=8, norm_layer=partial(nn.LayerNorm, eps=1e-6),  dropout=0.1,
#                  init_values=0.):
#         super().__init__()

#         self.scale_norm = norm_layer(d_model)
#         self.scale_attn = PoolingAttention(d_model, num_heads=n_heads, win_num=2, qkv_bias=True, pool_ratios=[4, 2, 1])
#         self.drop_path = nn.Dropout(dropout)

#         self.query_norm = norm_layer(d_model)
#         self.feat_norm = norm_layer(d_model)
#         self.attn = nn.MultiheadAttention(d_model, n_heads,dropout=dropout)
#         self.gamma = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

#     def forward(self, query,feat):
#             #query:l,b,d;feat:l,b,d
#         def _inner_forward(query, feat):

#             query = self.drop_path(self.scale_attn(self.scale_norm(query))) + query

#             attn = self.attn(self.query_norm(query),
#                              self.feat_norm(feat),self.feat_norm(feat))[0]
#             return query + self.gamma * attn
#         query = _inner_forward(query, feat)
#         return query

# class InteractionBlock(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.injector = Injector(d_model=d_model)

#     def forward(self,x,xs):
#         x = self.injector(x.permute(1,0,2),xs.permute(1,0,2)).permute(1,0,2)
#         return x

# class Focus_Neck(nn.Module):
#     def __init__(self, d_model=192, n_layers=1):
#         super().__init__()
#         self.d_model = d_model
#         self.n_layers = n_layers
#         self.interactions = nn.ModuleList(
#             [InteractionBlock(d_model=d_model)
#             for _ in range(n_layers)
#         ])

#     def forward(self, x, xs):
#         #  x : (B, L, D)
#         #  caches : [cache(layer) for all layers], cache : (h, inputs)

#         #  y : (B, L, D)
#         #  caches : [cache(layer) for all layers], cache : (h, inputs)

#         for i in range(self.n_layers):
#             x = self.interactions[i](x, xs)
#         return x