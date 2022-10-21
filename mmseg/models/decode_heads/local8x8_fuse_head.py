###########################################################################
# Created by: pmy
# Copyright (c) 2019
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.vit import Block

from mmcv.cnn import build_norm_layer


class SFTLayer(nn.Module):
    def __init__(self, head_channels):
        super(SFTLayer, self).__init__()

        self.SFT_scale_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_scale_conv1 = nn.Conv2d(head_channels, head_channels, 1)

        self.SFT_shift_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_shift_conv1 = nn.Conv2d(head_channels, head_channels, 1)

    def forward(self, local_features, global_features):
        #print('=====local_features=====global_features=====')
        #print(local_features.shape, global_features.shape)
        scale = self.SFT_scale_conv1(F.relu(self.SFT_scale_conv0(global_features),inplace=True))
        shift = self.SFT_shift_conv1(F.relu(self.SFT_shift_conv0(global_features),inplace=True))
        fuse_features = local_features * (scale+1) +shift
        return fuse_features

@HEADS.register_module()
class Local8x8_fuse_head(BaseDecodeHead):

    #def __init__(self, img_size=320, mla_channels=128, mlahead_channels=64,
    def __init__(self, img_size=320, mla_channels=128, mlahead_channels=128,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(Local8x8_fuse_head, self).__init__(**kwargs)

        self.img_size = img_size
        self.channels = mla_channels
        self.head_channels = mlahead_channels
        self.norm_cfg = norm_cfg
        self.BatchNorm = norm_layer

        self.SFT_head = SFTLayer(self.head_channels)
        self.edge_head = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels), nn.ReLU(),
            nn.Conv2d(self.head_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, local_features, global_features):
        fuse_features = self.SFT_head(local_features, global_features)
        fuse_edge = self.edge_head(fuse_features)
        fuse_edge = torch.sigmoid(fuse_edge)
        return fuse_edge, fuse_features