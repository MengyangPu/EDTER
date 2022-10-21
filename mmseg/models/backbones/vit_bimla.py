import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import BACKBONES

from mmcv.cnn import build_norm_layer


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_base_p16_224-4e355ebd.pth',
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        pretrained_finetune='pretrain/jx_vit_base_p16_384-83fb41ba.pth'),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        pretrained_finetune='pretrain/jx_vit_large_p16_384-b3be5167.pth'),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
    'deit_base_distilled_path16_384': _cfg(
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        pretrained_finetune='pretrain/deit_base_distilled_patch16_384.pth'
    )
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape  #(1,3,512,512)
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x) #1,1024,32,32
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Conv_BIMLA(nn.Module):
    def __init__(self, in_channels=1024, mla_channels=256, norm_cfg=None):
        super(Conv_BIMLA, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())

        self.mla_b2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_b3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_b4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_b5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())

    def to_2D(self, x):
        n, hw, c = x.shape
        h=w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)
        return x

    def forward(self, res2, res3, res4, res5):

        res2 = self.to_2D(res2)  #[1, 1024, 32, 32]
        res3 = self.to_2D(res3)  #[1, 1024, 32, 32]
        res4 = self.to_2D(res4)  #[1, 1024, 32, 32]
        res5 = self.to_2D(res5)  #[1, 1024, 32, 32]

        mla_p5_1x1 = self.mla_p5_1x1(res5)
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p4_plus = mla_p5_1x1 + mla_p4_1x1
        mla_p3_plus = mla_p4_plus + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p5 = self.mla_p5(mla_p5_1x1)
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        mla_b2_plus = mla_p2_1x1
        mla_b3_plus = mla_b2_plus + mla_p3_1x1
        mla_b4_plus = mla_b3_plus + mla_p4_1x1
        mla_b5_plus = mla_b4_plus + mla_p5_1x1

        mla_b2 = self.mla_b2(mla_b2_plus)
        mla_b3 = self.mla_b3(mla_b3_plus)
        mla_b4 = self.mla_b4(mla_b4_plus)
        mla_b5 = self.mla_b5(mla_b5_plus)

        return mla_b2, mla_b3, mla_b4, mla_b5, mla_p2, mla_p3, mla_p4, mla_p5


@BACKBONES.register_module()
class VIT_BIMLA(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, model_name='vit_large_patch16_384', img_size=384, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
                 num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None, 
                 pos_embed_interp=False, random_init=False, align_corners=False, mla_channels=256, 
                 mla_index=(5,11,17,23), **kwargs):
        super(VIT_BIMLA, self).__init__(**kwargs)
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        self.norm_layer = norm_layer
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners
        self.mla_channels = mla_channels
        self.mla_index = mla_index

        self.num_stages = self.depth
        self.out_indices= tuple(range(self.num_stages))

        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone, img_size=self.img_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))   #[1, 1, 1024]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))  #[1, 1025, 1024]
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)])

        self.mla = Conv_BIMLA(in_channels=self.embed_dim, mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)

        self.norm_0 = norm_layer(self.embed_dim)
        self.norm_1 = norm_layer(self.embed_dim)
        self.norm_2 = norm_layer(self.embed_dim)
        self.norm_3 = norm_layer(self.embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        # nn.init.normal_(self.pos_embed, std=0.02)
        # nn.init.zeros_(self.cls_token)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.random_init == False:
            self.default_cfg = default_cfgs[self.model_name]

            if self.model_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp, num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, filter_fn=self._conv_filter)
            else:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp, num_patches=self.patch_embed.num_patches, align_corners=self.align_corners)
        else:
            print('Initialize weight randomly')

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    def forward(self, x):
        B = x.shape[0] #B=1
        x = self.patch_embed(x) #[1, 1024, 32, 32]
        x = x.flatten(2).transpose(1, 2) #([1, 1024, 1024])

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks [1, 1, 1024]
        x = torch.cat((cls_tokens, x), dim=1)  #[1, 1025, 1024]
        x = x + self.pos_embed  #[1, 1025, 1024]
        x = x[:,1:]   #[1, 1024, 1024]
        x = self.pos_drop(x)
              
        outs = []  #len(outs) = 24
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)

        c6 = self.norm_0(outs[self.mla_index[0]])   #[1, 1024, 1024] mla_index=(5, 11, 17, 23)
        c12 = self.norm_1(outs[self.mla_index[1]])  #[1, 1024, 1024]
        c18 = self.norm_2(outs[self.mla_index[2]])  #[1, 1024, 1024]
        c24 = self.norm_3(outs[self.mla_index[3]])  #[1, 1024, 1024]

        b6, b12, b18, b24, p6, p12, p18, p24 = self.mla(c6, c12, c18, c24)
        
        return (b6, b12, b18, b24, p6, p12, p18, p24)

