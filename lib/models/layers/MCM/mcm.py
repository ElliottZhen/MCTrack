
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.models.mctrack.utils import combine_tokens
from lib.models.layers.MCM.gaussianimage_rs import GaussianImage_RS as Gaborsplatting
from lib.models.layers.MCM.focus_neck import Focus_Neck
from lib.models.layers.MCM.mamba import Mamba_Neck


class SelectToken(nn.Module):
    def __init__(self, topk=3, win_size=2):
        super().__init__()
        self.topk = topk
        self.win_size = win_size
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.gabor_splatting = Gaborsplatting(num=16, h=4, w=4)

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
        x = x.permute(0,2,1).reshape(B,C,h_s,h_s)
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
    


class MCM(nn.Module):
    def __init__(self, topk=3, win_size=2, dim=192, nlayer=1, state=16):
        super().__init__()
        self.topk_num = topk
        self.win_size = win_size
        self.dim = dim
        self.n_layers = nlayer
        self.d_state = state
        self.select_token = SelectToken(topk=self.topk_num, win_size=self.win_size)
        self.focus = Focus_Neck(d_model=self.dim, n_layers=self.n_layers)
        self.mamba = Mamba_Neck(in_channel=self.dim, d_model=self.dim, d_inner=2*self.dim,
                                      n_layers=self.n_layers, dt_rank=self.dim//16, d_state=self.d_state)
        self.pos_drop = nn.Dropout(p=0.)
        
    def forward(self, z, x, hidden_state):

        lens_z = z.size[1]
        lens_x = x.size[1]
        x_selected = self.select_token(z, x)
        x = self.focus(x, x_selected)

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = self.pos_drop(x)

        xs = x[:, lens_z:lens_z + lens_x]
        x, hidden_state = self.mamba(x, xs, hidden_state)

        return x, hidden_state