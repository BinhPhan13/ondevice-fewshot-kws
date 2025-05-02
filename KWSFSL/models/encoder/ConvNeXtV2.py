# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(dim)
        self.pwconv2 = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.Conv2d(in_chans, dims[0], kernel_size=2, padding= 2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m): 
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
#         print(x.size()); exit()
        return x

def convnextv2_atto_tung(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[16, 32, 64, 128], **kwargs)       # 3482 MB ~0.444 (0) -> 0.270 (9)
#     model = ConvNeXtV2(depths=[2, 2, 3, 2], dims=[16, 32, 64, 128], **kwargs)       # 3146 MB ~0.453 (0) -> 0.275 (9) 
#     model = ConvNeXtV2(depths=[2, 2, 3, 2], dims=[32, 64, 128, 256], **kwargs)      # 5688 MB ~0.436 (0) -> 0.256 (9)
#     model = ConvNeXtV2(depths=[2, 2, 3, 2], dims=[32, 64, 128, 96], **kwargs)       # 5660 MB ~0.439 (0) -> 0.262 (9)
#     model = ConvNeXtV2(depths=[1, 1, 3, 1], dims=[32, 64, 128, 96], **kwargs)       # 4280 MB ~0.444 (0) -> 0.268 (9)
#     model = ConvNeXtV2(depths=[2, 2, 4, 2], dims=[32, 64, 128, 96], **kwargs)       # 5882 MB ~0.448 (0) -> 0.260 (9)
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs) # Atto 7984 MB ~0.429 (0) -> 0.249 (9) -> 0.241 (32)
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[32, 64, 128, 256], **kwargs)      # 6372 MB ~0.434 (0) -> 0.252 (9)
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[16, 32, 64, 256], **kwargs)       # 3608 MB ~0.436 (0) -> 0.263 (9)
    model = ConvNeXtV2(depths=[2, 2, 9, 2], dims=[16, 32, 64, 256], **kwargs)       # 3610 MB ~0.438 (0) -> 0.262 (9)
    return model

def convnextv2_virus0(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 2, 2], dims=[64, 128, 256, 512], **kwargs) 
    return model    
    
def convnextv2_virus(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 2, 2], dims=[34, 69, 138, 276], **kwargs) 
    return model
    
    
def convnextv2_small(**kwargs):
    model = ConvNeXtV2(depths=[1, 1, 2, 1], dims=[28, 16, 32, 64], **kwargs)
    return model  
    
def convnextv2_ronto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 3, 2], dims=[20, 40, 80, 160], **kwargs)
    return model    
    
def convnextv2_yokto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 3, 2], dims=[25, 50, 100, 200], **kwargs)
    return model    

def convnextv2_zepto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[30, 60, 120, 240], **kwargs)
    return model    
    
def convnextv2_zepto_1(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[30, 69, 138, 276], **kwargs)
    return model    

def convnextv2_atto_0(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 276], **kwargs)
    return model

def convnextv2_atto_0_5(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[53, 92, 160, 276], **kwargs)
    return model

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
