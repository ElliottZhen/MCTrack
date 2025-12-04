import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from lib.models.layers.MCM.gaussianimage_rs import GaussianImage_RS
from lib.models.layers.MCM.focus_neck import Focus_Neck

class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()
        
        #self.dim = dim
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim, bias=True)

        self.alpha = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, feat):

        feat = feat.permute(0, 3, 1, 2).contiguous()
        out = feat * self.alpha[0] + self.conv(feat) * self.alpha[1]
        out = out.permute(0, 2, 3, 1).contiguous()

        return out


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

        # self.state_fusion = StateFusion(self.d_inner * self.d_state)

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

        # b_h, l_h, d_h, n_h = h.shape

        # height = int(math.sqrt(l_h))
        # h = h.flatten(2).permute(0, 2, 1).view(b_h, -1, height, height).contiguous()

        # h = rearrange(h, "b (h w) d n -> b (d n) h w", h=int(math.sqrt(l_h)), w=int(math.sqrt(l_h))).contiguous()
        # h = self.state_fusion(h)
        # h = rearrange(h, "b (d n) h w -> b (h w) d n", d=d_h, n=n_h).contiguous()

        # h = h.flatten(2).permute(0, 2, 1).view(b_h, l_h, d_h, n_h).contiguous()

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

class Mamba_Neck(nn.Module):
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
        self.small_windows = 8
        self.layers = nn.ModuleList(
            [ResidualBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
             for _ in range(n_layers)])
        self.interactions = nn.ModuleList(
            [InteractionBlock(d_model=d_model)
            for _ in range(n_layers)
        ])

        # self.state_fusion = StateFusion(self.d_model)


    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C).contiguous()
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C).contiguous()
        return windows


    def window_reverse(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1).contiguous()
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).contiguous()
        return x

    def forward(self, x, xs, h):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs

        for i in range(self.n_layers):

            B, N, C = xs.shape[0], xs.shape[1], xs.shape[2]
            xs_windows = xs.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
            xs_windows = self.window_partition(xs_windows, self.small_windows)

            # xs_windows = self.state_fusion(xs_windows)

            xs_windows = xs_windows.view(-1, self.small_windows * self.small_windows, C).contiguous()  # nW*B, window_size*window_size, C

            xs_windows, h = self.layers[i](xs_windows, h)

            xs_windows = xs_windows.view(-1, self.small_windows, self.small_windows, C).contiguous()
            xs = self.window_reverse(xs_windows, self.small_windows, int(math.sqrt(N)), int(math.sqrt(N))).flatten(1, 2).contiguous()

            x = self.interactions[i](x, xs)

        return x, h

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
    


class SelectToken(nn.Module):
    def __init__(self, topk=3, win_size=2):
        super().__init__()
        self.topk = topk
        self.win_size = win_size
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.gabor_splatting = GaussianImage_RS()

    def naive_xcorr(self, z, x):
        # naive cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
    
    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size, window_size, C)
        windows = windows.flatten(2, 3)
        return windows

    def forward(self, z, x):
        B, N_t, C = z.shape
        h_t = int(math.sqrt(N_t))
        z = z.permute(0,2,1).reshape(B,C,h_t,h_t)
        z_max = self.maxpool(z)

        B, N_s, C = x.shape
        h_s = int(math.sqrt(N_s))
        x = x.permute(0,2,1).reshape(B,C,h_s,h_s).contiguous()
        x_windows = self.window_partition(x, self.win_size)  # b num_win n_win c

        response_map = self.naive_xcorr(z_max, x)  # b 1 h w

        windows = self.window_partition(response_map, self.win_size)  # b num_win n_win 1
        windows_mean = torch.mean(windows, dim=2, keepdim=True)  # b num_win 1 1
        index_windows = torch.topk(windows_mean,k=self.topk,dim=1)[1] 

        index_windows = index_windows.expand(-1,-1,x_windows.size(2),x_windows.size(3))
        x_selected = torch.gather(x_windows, dim=1, index=index_windows)

        x_selected_win = x_selected.view(B, self.topk, self.win_size, self.win_size, C)
        x_selected_win = x_selected_win.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, self.win_size, self.win_size)

        x_selected_up = F.interpolate(x_selected_win, scale_factor=2, mode='bilinear', align_corners=False)  # b c h w
        
        res_feat = (self.gabor_splatting(x_selected_up) + x_selected_up).reshape(B, self.topk, C, -1)
        res_feat = res_feat.permute(0, 1, 3, 2).reshape(B, -1, C)

        return res_feat