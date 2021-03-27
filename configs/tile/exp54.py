# model settings
model = dict(
    type='ATSS',  # universeNet
    pretrained=(
        'https://github.com/shinya7y/UniverseNet/releases/download/20.06/'
        'res2net50_v1b_26w_4s-3cf99910_mmdetv2.pth'),
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='SEPC',
            out_channels=256,
            stacked_convs=4,
            pconv_deform=True,
            lcconv_deform=True,
            ibn=False,  # please set imgs/gpu >= 4
        )
    ],
    bbox_head=dict(
        type='ATSSSEPCHead',
        #num_classes=80,
        num_classes=8,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            #ratios=[1.0],
            ratios=[0.5, 1.0, 2.0],  # clw modify
            octave_base_scale=2,
            #octave_base_scale=[1, 8],  # clw modify
            #scales_per_octave=1,
            scales_per_octave=3,
            #scales_per_octave=6,  # clw modify
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_cls=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # clw modify
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),

        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg = dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))




data = dict(
    samples_per_gpu=2,
    #samples_per_gpu=4,
    #samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='TileDataset',
        #type='TileMosaicDataset',
        #ann_file='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_2k.json',
        ann_file='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train.json',
        #img_prefix='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs_2k',
        img_prefix='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            #dict(type='LoadTemplate', template_path='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_template_imgs'),

            #dict(type='LoadMosaicImageAndAnnotations', with_bbox=True, with_mask=False, image_shape=[2048, 2048],
            #dict(type='LoadMosaicImageAndAnnotations', with_bbox=True, with_mask=False, image_shape=[4096, 4096],
            #     hsv_aug=False, h_gain=0.014, s_gain=0.68, v_gain=0.36, skip_box_w=1, skip_box_h=1, template_path='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_template_imgs_2k'),

            #dict(type='GtBoxBasedCrop', crop_size=(1024, 1024)),  # clw modify
            #dict(type='Resize', img_scale=(2048, 2048), keep_ratio=True),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            #dict(type='ConcatTemplate'),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='TileDataset',
        ann_file='/home/user/dataset/2021tianchi/tile_round2_train_20210208/val.json',
        img_prefix='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            #dict(type='LoadTemplate', template_path='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_template_imgs'),
            dict(
                type='MultiScaleFlipAug',
                #scale_factor=1.0,
                #img_scale=(2048, 2048),
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    #dict(type='ConcatTemplate'),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='TileDataset',
        ann_file='/home/user/dataset/2021tianchi/tile_round2_train_20210208/val.json',
        img_prefix='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            #dict(type='LoadTemplate', template_path='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_template_imgs'),
            dict(
                type='MultiScaleFlipAug',
                #scale_factor=1.0,
                #img_scale=(2048, 2048),
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    #dict(type='ConcatTemplate'),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
#evaluation = dict(interval=1, metric='bbox')
evaluation = dict(interval=1, metric=['mAP','bbox'])
optimizer = dict(type='SGD', lr=0.02/16 * data['samples_per_gpu'], momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    #warmup_iters=500,
    warmup_iters=1000,  # clw modify
    warmup_ratio=0.001,
    step=[8, 11])
    # step=[16, 23])
    # step=[28, 34])
total_epochs = 12
# total_epochs = 24
# total_epochs = 36
checkpoint_config = dict(interval=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = '/home/user/.cache/torch/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15_concat6.pth'
load_from = '/home/user/.cache/torch/mmdetection/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_23-f9f426a3.pth'
#load_from = '/home/user/.cache/torch/mmdetection/r50-FPN-1x_classsampling_publish.pth'  # 2.20 modify
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.0)
work_dir = './work_dirs/exp54'
gpu_ids = range(0, 1)
