_base_ = [
    '../../_base_/models/ams_r50.py', '../../_base_/default_runtime.py'
]

'''# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/ouyangjun/workspace/data/a/songhao/ViFi-CLIP/videos/UCF101'
data_root_val = '/home/ouyangjun/workspace/data/a/songhao/ViFi-CLIP/videos/UCF101'
ann_file_train = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/ucf101.txt'
ann_file_val = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/ucf101.txt'
ann_file_test = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/ucf101.txt'''

dataset_type = 'RawframeDataset'
data_root = '/home/ouyangjun/workspace/data/a/workspace/MOMA-LRG/videos/frames_myself/all_frames/'
data_root_val = '/home/ouyangjun/workspace/data/a/workspace/MOMA-LRG/videos/frames_myself/all_frames/'
ann_file_train = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_train.txt'
ann_file_val = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_test.txt'
ann_file_test = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA-LRG/data/MOMA/rewrite_test.txt'

#dataset_type = 'RawframeDataset'
#data_root = '/home/ouyangjun/workspace/MOMA/MOMA-1.0/all_frames/'
#data_root_val = '/home/ouyangjun/workspace/MOMA/MOMA-1.0/all_frames/'
#ann_file_train = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA/data/MOMA/rewrite_train.txt'
#ann_file_val = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA/data/MOMA/rewrite_test.txt'
#ann_file_test = '/home/ouyangjun/workspace/data/a/sx/mmaction2_MOMA/data/MOMA/rewrite_test.txt'


model = dict(backbone=dict(num_segments=8, gamma=1),
             neck=dict(gamma=1),
             test_cfg=dict(average_clips='prob', fcn_test=True))

'''img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        twice_sample=True,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]


data = dict(
    videos_per_gpu=8,
    workers_per_gpu=1,
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.012, momentum=0.9, weight_decay=0.0005,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=20, 0hnorm_type=2))
# learning policy
lr_config = dict(policy='step', step=[75, 95])
total_epochs = 110
'''
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
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
        clip_len=1,
        frame_interval=1,
        num_clips=8,
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
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
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
    batch_size=16,
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
    batch_size=16,
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
    type='EpochBasedTrainLoop', max_epochs=110, val_begin=1, val_interval=1)
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
# runtime settings
work_dir = '/home/ouyangjun/workspace/data/a/songhao/mmaction2_MOMALRG/work_dirs/ams_r50_1x1x8_110e_sthv1_rgb_MOMA-LRG'


