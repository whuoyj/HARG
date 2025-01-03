_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=124,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))
'''
# dataset settings
dataset_type = 'VideoDataset'
data_root_val = '/home/ouyangjun/workspace/data/a/songhao/ViFi-CLIP/videos/UCF101/'
ann_file_test = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA/data/MOMA/test.txt'

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = dict(type='AccMetric')

test_cfg = dict(type='TestLoop')
'''


# dataset settings
# dataset_type = 'RawframeDataset'
# data_root = '/home/ouyangjun/workspace/data/a/workspace/MOMA-LRG/videos/frames_myself/all_frames/'
# data_root_val = '/home/ouyangjun/workspace/data/a/workspace/MOMA-LRG/videos/frames_myself/all_frames/'
# ann_file_train = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_train.txt'
# ann_file_val = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_test.txt'
# ann_file_test = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_test.txt'

# dataset_type = 'RawframeDataset'
# data_root = '/home/ouyangjun/workspace/MOMA/MOMA-1.0/all_frames/'
# data_root_val = '/home/ouyangjun/workspace/MOMA/MOMA-1.0/all_frames/'
# ann_file_train = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA/data/MOMA/rewrite_train.txt'
# ann_file_val = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA/data/MOMA/rewrite_test.txt'
# ann_file_test = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA/data/MOMA/rewrite_test.txt'
dataset_type = 'RawframeDataset'
data_root = '/home/ouyangjun/workspace/data/a/workspace/MOMA-LRG/videos/frames_myself/all_frames/'
data_root_val = '/home/ouyangjun/workspace/data/a/workspace/MOMA-LRG/videos/frames_myself/all_frames/'
ann_file_train = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_train.txt'
ann_file_val = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_test.txt'
ann_file_test = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_test.txt'


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        filename_tmpl='{:05}.jpg',
        data_prefix=dict(img=data_root_val),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        filename_tmpl='{:05}.jpg',
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        filename_tmpl='{:05}.jpg',
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)