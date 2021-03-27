norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='SingleStageDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='RepPointsV2Head',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        shared_stacked_convs=1,
        first_kernel_size=3,
        kernel_size=1,
        corner_dim=64,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=0.25),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_sem=dict(
            type='SEPFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.1),
        transform_method='exact_minmax'),

    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssignerV2', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        heatmap=dict(
            assigner=dict(type='PointHMAssigner', gaussian_bump=True, gaussian_iou=0.7),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.6),
        max_per_img=100))

data = dict(
    samples_per_gpu=2,
    #samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type='TileDataset',
        #type='TileMosaicDataset',
        ann_file=
        '/home/user/dataset/2021tianchi/tile_round2_train_20210208/train.json',
        #'/home/user/dataset/2021tianchi/tile_round2_train_20210208/train--toy.json',
        img_prefix=
        '/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            #dict(type='LoadMosaicImageAndAnnotations', with_bbox=True, with_mask=False, image_shape=[2048, 2048], hsv_aug=False, h_gain=0.014, s_gain=0.68, v_gain=0.36, skip_box_w=1, skip_box_h=1),
            #dict(type='GtBoxBasedCrop', crop_size=(1024, 1024)),  # clw modify
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            # dict(type='DefaultFormatBundle'),
            # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            dict(type='LoadRPDV2Annotations', num_classes=8),
            dict(type='RPDV2FormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights']),
        ]),
    val=dict(
        type='TileDataset',
        ann_file='/home/user/dataset/2021tianchi/tile_round2_train_20210208/val.json',
        img_prefix='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                #scale_factor=1.0,
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
            dict(
                type='MultiScaleFlipAug',
                #scale_factor=1.0,
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
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
#evaluation = dict(interval=1, metric='bbox')
evaluation = dict(interval=1, metric=['mAP','bbox'])
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
    # step=[16, 23])
total_epochs = 12
# total_epochs = 24
checkpoint_config = dict(interval=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = '/home/user/.cache/torch/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15_concat6.pth'
#load_from = '/home/user/.cache/torch/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
load_from = '/home/user/.cache/torch/mmdetection/reppoints_v2_r50_fpn_giou_mstrain_2x_coco-889d053a.pth'
#load_from = '/home/user/.cache/torch/mmdetection/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth'
#load_from = '/home/user/.cache/torch/mmdetection/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
#load_from = '/home/user/.cache/torch/mmdetection/r50-FPN-1x_classsampling_publish.pth'
resume_from = None
workflow = [('train', 1)]
#fp16 = dict(loss_scale=512.0)
work_dir = './work_dirs/exp46'
gpu_ids = range(0, 1)
