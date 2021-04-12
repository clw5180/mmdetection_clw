model = dict(
    type='DualCascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        # plugins=[
        #     dict(
        #         cfg=dict(
        #             type='GeneralizedAttention',
        #             spatial_range=-1,
        #             num_heads=8,
        #             attention_type='0010',
        #             kv_stride=2),
        #         stages=(False, False, True, True),
        #         position='after_conv2')
        #     ],
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        ### clw modify
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),  # c3-c5
        ###
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        # type='FPN_CARAFE',
        # in_channels=[256, 512, 1024, 2048],
        # out_channels=256,
        # num_outs=5,
        # start_level=0,
        # end_level=-1,
        # norm_cfg=None,
        # act_cfg=None,
        # order=('conv', 'norm', 'act'),
        # upsample_cfg=dict(
        #     type='carafe',
        #     up_kernel=5,
        #     up_group=1,
        #     encoder_kernel=3,
        #     encoder_dilation=1,
        #     compressed_channels=64)),
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1, 8],
            ratios=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            #add_context=True,  # clw modify
            add_context=False,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            # assigner=dict(
                # type='MaxIoUAssigner',
                # #pos_iou_thr=0.7,
                # pos_iou_thr=0.5,   # clw modify
                # neg_iou_thr=0.3,
                # min_pos_iou=0.3,
                # match_low_quality=True,
                # ignore_iof_thr=-1),

            # modify from zhengye
            assigner=dict(type='ATSSAssigner', topk=9),

            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.3,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.4,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=100)))

# albu_train_transforms = [
#     dict(type='RandomRotate90', always_apply=False, p=0.5),
#     dict(type='Transpose', always_apply=False, p=0.5),
# ]

data = dict(
    samples_per_gpu=1,
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
            dict(
                type='LoadTemplate',
                template_path=
                '/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_template_imgs'
            ),
            dict(
                type='Resize',
                img_scale=[(4096, 1536), (4096, 1800)],
                multiscale_mode='range',
                keep_ratio=True),
            #dict(type='LoadMosaicImageAndAnnotations', with_bbox=True, with_mask=False, image_shape=[2048, 2048], hsv_aug=False, h_gain=0.014, s_gain=0.68, v_gain=0.36, skip_box_w=1, skip_box_h=1),
            #dict(type='GtBoxBasedCrop', crop_size=(1024, 1024)),  # clw modify
            dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical']),
            dict(type='BboxesJitter', shift_ratio=0.1),
            # dict(
            #     type='Albu',
            #     transforms=albu_train_transforms,
            #     bbox_params=dict(
            #         type='BboxParams',
            #         format='pascal_voc',
            #         label_fields=['gt_labels'],
            #         min_visibility=0.0,
            #         filter_lost_elements=True),
            #     keymap={
            #         'img': 'image',
            #         'gt_bboxes': 'bboxes'
            #     },
            #     update_pad_shape=False,
            #     skip_img_without_anno=True),
            # dict(type='CutOut', n_holes=(8, 16), cutout_shape=(64, 64)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ConcatTemplate'),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        filter_empty_gt=False),  # clw note: add negative sample
    val=dict(
        type='TileDataset',
        ann_file='/home/user/dataset/2021tianchi/tile_round2_train_20210208/val.json',
        img_prefix='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTemplate', template_path='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_template_imgs'),
            dict(
                type='MultiScaleFlipAug',
                #scale_factor=1.0,
                img_scale=(1536, 1536),
                #img_scale=(1024, 1024),
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
                    dict(type='ConcatTemplate'),
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
            dict(type='LoadTemplate',
                 template_path='/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_template_imgs'),
            dict(
                type='MultiScaleFlipAug',
                #scale_factor=1.0,
                img_scale=(1536, 1536),
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
                    dict(type='ConcatTemplate'),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
#evaluation = dict(interval=1, metric='bbox')
evaluation = dict(interval=1, metric=['mAP','bbox'])
optimizer = dict(type='SGD', lr= 0.02 * data['samples_per_gpu'] / 16, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    #step=[8, 11])
    step=[16, 19])
    # step=[16, 23])
#total_epochs = 12
total_epochs = 20
# total_epochs = 24
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/user/.cache/torch/mmdetection/htc_r50_fpn_20e_coco_20200319-fe28c577.pth'
#load_from = '/home/user/.cache/torch/mmdetection/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202-42e767a2.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/exp56'
gpu_ids = range(0, 1)

fp16 = dict(loss_scale=512.0)