from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
# from utils import *
# from .utils import *
import torch
import torch.nn as nn
import numpy as np
import math

class GaussianImage_RS(nn.Module):
    def __init__(self, num, h, w):
        super().__init__()
        
        self.dim = 192
        self.init_num_points = num
        self.H, self.W = h, w
        self.BLOCK_W, self.BLOCK_H = self.H // 2, self.W // 2
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = "cuda"
        # self.device = "cpu"

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))

        # ========== 新增：Gabor参数 ==========
        self._frequency = nn.Parameter(torch.rand(self.init_num_points, 2))  # 频率参数 (u₀, v₀)
        self._phase = nn.Parameter(torch.rand(self.init_num_points, 1))      # 相位参数 φ

        # # 修改初始化：频率初始化为较小值，相位初始化为0
        # self._frequency = nn.Parameter(torch.randn(self.init_num_points, 2) * 0.01)  # 使用较小的初始值
        # # self._phase = nn.Parameter(torch.zeros(self.init_num_points, 1))  # 初始相位为0

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3, device=self.device)
        self.rotation_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))

        self.proj_up = nn.Linear(3, self.dim)

    @property
    def get_scaling(self):
        return torch.abs(self._scaling+self.bound)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)*2*math.pi
    
    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity 
    
    # ========== 新增：频率和相位的属性 ==========
    @property
    def get_frequency(self):
        # return self._frequency
        # 使用softplus确保频率为正，同时避免梯度消失
        return torch.nn.functional.softplus(self._frequency, beta=10)  # beta=10使得在0附近接近线性，但大于0
    
    @property  
    def get_phase(self):
        return torch.sigmoid(self._phase) * 2 * math.pi  # 映射到 [0, 2π]

    def forward(self, x):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(self.get_xyz, self.get_scaling, self.get_rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity, self.get_frequency, self.get_phase, 
                self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        # print(out_img.device)
        out_img = torch.clamp(out_img, 0, 1).reshape(-1, self.H * self.W, 3) #[H, W, 3]
        x_out = self.proj_up(out_img).permute(0,2,1).reshape(-1, x.shape[1], self.H, self.W)
        return x_out
    
if __name__ == "__main__":
    input = torch.randn(1, 3, 4, 4).cuda()
    model = GaussianImage_RS().cuda()
    out = model(input)
    print(out)
    print(out.shape)
