_base_ = [
    'models/cascade_rcnn_r50_fpn.py',
    'datasets/tile_detection.py',
    'schedules/schedule_1x.py',
    'default_runtime.py'
]

# fp16 settings
fp16 = dict(loss_scale=512.)