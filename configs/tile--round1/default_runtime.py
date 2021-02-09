checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    #interval=10,  # clw modify
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = None
#load_from = '/home/user/.cache/torch/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
#load_from = '/home/user/.cache/torch/mmdetection/cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth'
#load_from = '/home/user/.cache/torch/mmdetection/htc_r50_fpn_20e_coco_20200319-fe28c577.pth'
load_from = '/home/user/.cache/torch/mmdetection/detectors_htc_r50_1x_coco-329b1453.pth'
#load_from = '/home/user/.cache/torch/mmdetection/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth'
resume_from = None
workflow = [('train', 1)]
