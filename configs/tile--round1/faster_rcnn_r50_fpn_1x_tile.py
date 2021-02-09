_base_ = [
    'models/faster_rcnn_r50_fpn_tile.py',
    'datasets/tile_detection.py',
    'schedules/schedule_1x.py',
    'default_runtime.py'
]

# fp16 settings
fp16 = dict(loss_scale=512.)