num_classes = 20
# model.py
model = dict(
    backbone=dict(
        type='LiteHRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1,
                strides=(2, 2),
                extra_stride=False,
                input_norm=False),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=((40, 80), (40, 80, 160), (40, 80, 160, 320))),
            out_modules=dict(
                conv=dict(enable=False, channels=320),
                position_att=dict(
                    enable=False,
                    key_channels=128,
                    value_channels=320,
                    psp_size=(1, 3, 6, 8)),
                local_att=dict(enable=False)),
            out_aggregator=dict(enable=False),
            add_input=False)),
    type='EncoderDecoder',  # change from ClassIncrEncoderDecoder
    pretrained=None,
    decode_head=dict(
        type='CustomFCNHead',
        in_channels=[40, 80, 160, 320],
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        channels=40,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        enable_aggregator=True,
        enable_out_norm=False,
        loss_decode=[
            dict(type='CrossEntropyLossWithIgnore', loss_weight=1.0)
        ]),
    # modified during runtime
    train_cfg = dict(
        max_loss=dict(
            mix_loss=dict(
                enable=False,
                weight=0.1
            )
        )
    ),
    test_cfg=dict(
        mode="whole",
        output_scale=3.0,
    ))
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnet18_imagenet1k_rsc.pth'
# fp16 = dict(loss_scale=512.0)


# added
log_level = 'INFO'  # The level of logging.
workflow = [('train', 1)]
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=True,  # Whether count by epoch or not.
    interval=1)  # The save interval.

optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='Adam',  # Type of optimizers, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
    lr=0.001,
    eps=1e-8,
    weight_decay=0.0)
# optimizer_config = dict(
#     type="Fp16OptimizerHook", grad_clip=dict(max_norm=40, norm_type=2),
#     distributed=False, loss_scale=512.0)  # Config used to build the optimizer hook, refer to 
optimizer_config = dict()
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=25
) # Total number of iterations. For EpochBasedRunner use `max_epochs`

# lr_config = dict(
#   policy='ReduceLROnPlateau', metric='mDice', patience=5, iteration_patience=0, interval=1, min_lr=1e-06,
#   warmup='linear', warmup_iters=100, warmup_ratio=0.3333333333333333
# )
# lr_config = dict(policy='CosineAnnealing', min_lr=1e-06, by_epoch=True, warmup='linear', warmup_iters=100, warmup_ratio=0.333333)
lr_config = dict(
    policy='poly',  # The policy of scheduler, also support Step, CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    power=0.9,  # The power of polynomial decay.
    min_lr=0.0001,  # The minimum learning rate to stable the training.
    by_epoch=False)  # Whether count by epoch or not.

log_config = dict(  # config to register logger hook
    interval=10,  # Interval to print the log
    hooks=[dict(type='TextLoggerHook', by_epoch=True),])
evaluation=dict(interval=1, metric='mDice', show_log=True, save_best='mDice', rule='greater')

resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the iteration when the checkpoint's is saved.

# data_pipeline.py
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=(544, 544),
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # dict(
    #     type='MaskCompose',
    #     prob=0.5,
    #     lambda_limits=(4, 16),
    #     keep_original=False,
    #     transforms=[dict(type='PhotoMetricDistortion')]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='RandomRotate', prob=0.5, degree=30, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(544, 544),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# dataset="kvasir_seg"
# dataset_size="full"
data = dict(
    samples_per_gpu=8,  # Batch size of a single GPU
    workers_per_gpu=0,  # Worker to pre-fetch data for each single GPU
    persistent_workers=False,
    train=dict(
        # type="KvasirSegDataset",
        # img_dir=f"/home/eunwoo/work/gpu_util_comp/data/{dataset}/{dataset_size}/img_dir/train",
        # ann_dir=f"/home/eunwoo/work/gpu_util_comp/data/{dataset}/{dataset_size}/ann_dir/train",
        type="PascalVOCDataset",
        img_dir=f"/home/eunwoo/work/data/voc/voc1.0/JPEGImages",
        split=f"/home/eunwoo/work/data/voc/voc1.0/ImageSets/Segmentation/train.txt",
        ann_dir=f"/home/eunwoo/work/data/voc/voc1.0/SegmentationClass",
        pipeline=train_pipeline,
    ),
    val=dict(
        # type="KvasirSegDataset",
        # img_dir=f"/home/eunwoo/work/gpu_util_comp/data/{dataset}/{dataset_size}/img_dir/val",
        # ann_dir=f"/home/eunwoo/work/gpu_util_comp/data/{dataset}/{dataset_size}/ann_dir/val",
        type="PascalVOCDataset",
        img_dir=f"/home/eunwoo/work/data/voc/voc1.0/JPEGImages",
        split=f"/home/eunwoo/work/data/voc/voc1.0/ImageSets/Segmentation/val.txt",
        ann_dir=f"/home/eunwoo/work/data/voc/voc1.0/SegmentationClass",
        pipeline=test_pipeline,
    ),
    test=dict(
        # type="KvasirSegDataset",
        # img_dir=f"/home/eunwoo/work/gpu_util_comp/data/{dataset}/{dataset_size}/img_dir/val",
        # ann_dir=f"/home/eunwoo/work/gpu_util_comp/data/{dataset}/{dataset_size}/ann_dir/val",
        type="PascalVOCDataset",
        img_dir=f"/home/eunwoo/work/data/voc/voc1.0/JPEGImages",
        split=f"/home/eunwoo/work/data/voc/voc1.0/ImageSets/Segmentation/val.txt",
        ann_dir=f"/home/eunwoo/work/data/voc/voc1.0/SegmentationClass",
        pipeline=test_pipeline,
    ))
