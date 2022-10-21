# optimizer
optimizer = dict(type='SGD', lr=1e-7, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-11, by_epoch=False)
# runtime settings
total_iters = 4000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=4000, metric='mIoU')