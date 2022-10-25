model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
    #samples_per_gpu=32,
    #samples_per_gpu=16,
    #samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CIFAR10',
        data_prefix='/mnt/lustre/share/datasets/disk1',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR10',
        data_prefix='/mnt/lustre/share/datasets/disk1',
        pipeline=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR10',
        data_prefix='/mnt/lustre/share/datasets/disk1',
        pipeline=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
#lr_config = dict(policy='step', step=[20, 60, 100, 150, 180])
lr_config = dict(policy='step', step=[40, 100, 150])
#lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
#dist_params = dict(backend='nccl')
dist_params = dict(backend='hccl')
log_level = 'DEBUG'
load_from = None
resume_from = None
workflow = [('train', 1)]
#work_dir = './work_dirs/resnet50_8xb16_cifar10'
gpu_ids = range(0, 1)
