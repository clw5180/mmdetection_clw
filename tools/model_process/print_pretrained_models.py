import torch

model_path = '/home/user/.cache/torch/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
state_dict = torch.load(model_path)['state_dict']

print('end!')