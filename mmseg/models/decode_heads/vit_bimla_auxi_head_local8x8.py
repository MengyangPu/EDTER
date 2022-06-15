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


@HEADS.register_module()
class VIT_BIMLA_AUXIHead_LOCAL8x8(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, **kwargs):
        super(VIT_BIMLA_AUXIHead_LOCAL8x8, self).__init__(**kwargs)
        self.img_size = img_size
        if self.in_channels==1024:
            self.aux_0 = nn.Conv2d(self.in_channels, 256, kernel_size=1, bias=False)
            self.aux_1 = nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False)
        elif self.in_channels==256:
            self.aux = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, self.in_channels, 4, stride=2, padding=1, bias=False),
                nn.ConvTranspose2d(self.in_channels, 1, 8, stride=4, padding=2, bias=False),
                #nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=False)
            )

    def to_2D(self, x):
        n, hw, c = x.shape
        h=w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)
        return x

    def forward(self, x):
        x = self._transform_inputs(x)
        if x.dim()==3:
            x = x[:,1:]
            x = self.to_2D(x)

        if self.in_channels==1024:
            x = self.aux_0(x)
            x = self.aux_1(x)
        elif self.in_channels==256:
            x = self.aux(x)
            x = torch.sigmoid(x)
        return x
