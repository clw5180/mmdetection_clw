import asyncio
from mmdet.apis import init_detector, inference_detector, async_inference_detector
import mmcv
from mmdet.utils.contextmanagers import concurrent

import torch
import os
from tqdm import tqdm
import json
import multiprocessing
from multiprocessing import Pool
import random
import cv2
import time
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
#config_file = 'configs/tile/faster_rcnn_r50_fpn_1x_tile.py'
config_file = '/home/user/mmdetection/work_dirs/exp21/exp21.py'
#checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_tile/epoch_12.pth'
checkpoint_file = '/home/user/mmdetection/work_dirs/exp21/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = ('1', '2', '3', '4', '5', '6', '7', '8')
#
#img_folder = '/home/user/dataset/tile_round1_testA_20201231/testA_imgs-'
img_folder = '/home/user/dataset/2021tianchi/tile_round2_train_20210208/train_imgs'
img_names = os.listdir(img_folder)

colors = [[random.randint(0, 255) for _ in range(3)] for _ in model.CLASSES]


async def main():
    start = time.time()

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device='cuda:0'))



    # 单进程
    share_submit_result = []
    for img_name in tqdm(img_names):
        img_path = os.path.join(img_folder, img_name)
        img = mmcv.imread(img_path)

        async with concurrent(streamqueue):
            result = await async_inference_detector(model, img)

        for cls, item in enumerate(result):
            if item is None:
                continue
            else:
                for row in item:
                    share_submit_result.append({'name': img_name, 'category': cls+1, 'bbox': row[:4].tolist(), 'score': str(row[4])})

        # save the visualization results to image files
        ##### model.show_result(img, result, out_file=img_name)
        for cls, item in enumerate(result):
            if item is None:
                continue
            else:
                for row in item:
                    label = '%s %.2f' % (model.CLASSES[int(cls)], row[4])
                    plot_one_box(row[:4], img, label=label, color=colors[cls], line_thickness=3)
        cv2.imwrite('output/' + img_name, img)

    # with open('result.json', 'w') as fp:
    #     json.dump(list(share_submit_result), fp, indent=4, ensure_ascii=False)

    print('time use: %.3fs' % (time.time() - start))


if __name__ == '__main__':
    asyncio.run(main())
