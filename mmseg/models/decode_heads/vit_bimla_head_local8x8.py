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


class BIMLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(BIMLAHead, self).__init__()
        self.head2_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )
        self.head3_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )
        self.head4_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )
        self.head5_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )
        self.head2 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )
        self.head3 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )
        self.head4 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )
        self.head5 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   )

    def forward(self, mla_b2, mla_b3, mla_b4, mla_b5, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        head5 = self.head5(mla_p5)
        
        head2_1 = self.head2_1(mla_b2)
        head3_1 = self.head3_1(mla_b3)
        head4_1 = self.head4_1(mla_b4)
        head5_1 = self.head5_1(mla_b5)
        return torch.cat([head2, head3, head4, head5, head2_1, head3_1, head4_1, head5_1], dim=1)



@HEADS.register_module()
class VIT_BIMLAHead_LOCAL8x8(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_BIMLAHead_LOCAL8x8, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.mlahead = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.local_features = nn.Sequential(
            nn.Conv2d(8 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.edge = nn.Conv2d(self.mlahead_channels, 1, 1)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7])
        x = self.local_features(x)
        edge = self.edge(x)
        edge = torch.sigmoid(edge)
        return edge, x
