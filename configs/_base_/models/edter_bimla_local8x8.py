# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder_LOCAL8x8',
    backbone=dict(
        type='VIT_BIMLA_LOCAL8x8',
        model_name='vit_base_patch16_384',
        img_size=768,
        patch_size=8,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=16,
        num_classes=1,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        mla_channels=256,
        mla_index=(2, 5, 8, 11)
        ),
    decode_head=dict(
        type='VIT_BIMLAHead_LOCAL8x8',
        in_channels=1024,
        channels=512,
        img_size=768,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')


