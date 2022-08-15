_base_ = [
    '../_base_/models/edter_bimla.py',
    '../_base_/datasets/nyud_hha.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(img_size=320,pos_embed_interp=True,drop_rate=0.,mla_channels=256,mla_index=(5,11,17,23)),
    decode_head=dict(img_size=320,mla_channels=256,mlahead_channels=64,num_classes=1),
    auxiliary_head=[
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=0,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=1,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=2,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=3,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True , loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=4,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=5,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=6,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
        type='VIT_BIMLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=7,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True , loss_weight=0.4)),
        ])

optimizer = dict(lr=1e-6, weight_decay=0.0002,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=10.)})
)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=False)

test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(280, 280))
find_unused_parameters = True
data = dict(samples_per_gpu=1)
