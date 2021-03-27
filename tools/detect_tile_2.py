from mmdet.apis import init_detector, inference_detector
import mmcv

import os
from tqdm import tqdm
import json
import multiprocessing
from multiprocessing import Pool
import random
import cv2
import time
import torch
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# Specify the path to model config and checkpoint file
config_file = 'configs/tile/faster_rcnn_r50_fpn_1x_tile.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_tile/epoch_10.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = ('1', '2', '3', '4', '5', '6')
#
img_folder = '/home/user/dataset/tile_round1_testA_20201231/testA_imgs-'
img_names = os.listdir(img_folder)

colors = [[random.randint(0, 255) for _ in range(3)] for _ in model.CLASSES]

# clw added
from pathlib import Path
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset
class LoadTileImages(Dataset):  # for inference
    def __init__(self, path):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if os.path.splitext(x)[-1].lower() in ['.bmp', '.jpg', '.jpeg', '.png']]
        ni = len(images)

        self.files = images
        self.nf = ni   # number of files
        self.mode = 'images'
        assert self.nf > 0, 'No images found in %s. Supported formats are:\nimages: %s' % (p, ['.bmp', '.jpg', '.jpeg', '.png'])

    def __getitem__(self, index):
        path = self.files[index]
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        # print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Convert
        img = img0.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite('letterbox.jpg', img.transpose(1, 2, 0)[:, :, ::-1])  # save letterbox image
        return path, img

    def __len__(self):
        return self.nf  # number of files

start = time.time()

dataset = LoadTileImages(img_folder)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=6,
                        pin_memory=True)

share_submit_result = []
for img_path, img in tqdm(dataloader):
    result = inference_detector(model, img_path)

    for cls, item in enumerate(result):
        if item is None:
            continue
        else:
            for row in item:
                share_submit_result.append({'name': img_path.split('/')[-1], 'category': cls+1, 'bbox': row[:4].tolist(), 'score': str(row[4])})


with open('result.json', 'w') as fp:
    json.dump(list(share_submit_result), fp, indent=4, ensure_ascii=False)

print('time use: %.3fs' % (time.time() - start))