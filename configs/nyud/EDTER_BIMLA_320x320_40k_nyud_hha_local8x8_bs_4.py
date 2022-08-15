_base_ = [
    '../_base_/models/edter_bimla_local8x8.py',
    '../_base_/datasets/nyud_hha.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(img_size=160, pos_embed_interp=True, drop_rate=0., mla_channels=256, mla_index=(2, 5, 8, 11)),
    decode_head=dict(img_size=160,mla_channels=256,mlahead_channels=64,num_classes=1),
    auxiliary_head=[
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=0,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=1,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=2,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=3,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True , loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=4,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=5,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=6,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead_LOCAL8x8',
        in_channels=256,
        channels=512,
        in_index=7,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True , loss_weight=0.4)),
        ],
    fuse_head=dict(
        type='Local8x8_fuse_head',
        in_channels=64,
        channels=64,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True , loss_weight=1.0))
)

optimizer = dict(lr=1e-6, weight_decay=0.0002,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=10.),'global_model': dict(lr_mult=0.),})
)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=False)

test_cfg = dict(mode='slide', crop_size=(160, 160), stride=(160, 160))
find_unused_parameters = True
data = dict(samples_per_gpu=1)
