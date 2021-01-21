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



def myfunc(idx, img_name, share_submit_result, share_lock ):
    print(idx)
    img_path = os.path.join(img_folder, img_name)
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    print('len of share_submit_result:', len(share_submit_result))

    for cls, item in enumerate(result):

        if item is None:
            continue
        else:
            for row in item:
                # 获取锁
                share_lock.acquire()
                share_submit_result.append(
                    {'name': img_name, 'category': cls + 1, 'bbox': row[:4].tolist(), 'score': str(row[4])})
                # 释放锁
                share_lock.release()



    # save the visualization results to image files
    ###### model.show_result(img, result, out_file=img_name)
    # for cls, item in enumerate(result):
    #     if item is None:
    #         continue
    #     else:
    #         for row in item:
    #             label = '%s %.2f' % (model.CLASSES[int(cls)], row[4])
    #             plot_one_box(row[:4], img, label=label, color=colors[cls], line_thickness=3)
    # cv2.imwrite(img_name, img)


if __name__ == '__main__':
    start = time.time()
    if not os.path.exists('output'):
        os.makedirs('output')

    # 字典声明方式
    # share_submit_result = multiprocessing.Manager().list()
    #
    # # 声明一个进程级共享锁
    # # 不要给多进程传threading.Lock()或者queue.Queue()等使用线程锁的变量，得用其进程级相对应的类
    # # 不然会报如“TypeError: can't pickle _thread.lock objects”之类的报错
    # share_lock = multiprocessing.Manager().Lock()
    #
    # p = Pool(12)
    # for idx, img_name in enumerate(img_names):
    #     p.apply_async(myfunc, args=(idx, img_name, share_submit_result, share_lock))
    #     #myfunc(idx, img_name, share_submit_result, share_lock)
    # p.close()
    # p.join()



    # 单进程
    share_submit_result = []
    for img_name in tqdm(img_names):
        img_path = os.path.join(img_folder, img_name)
        img = mmcv.imread(img_path)
        result = inference_detector(model, img)
        for cls, item in enumerate(result):
            if item is None:
                continue
            else:
                for row in item:
                    share_submit_result.append({'name': img_name, 'category': cls+1, 'bbox': row[:4].tolist(), 'score': str(row[4])})

        # save the visualization results to image files
        ###### model.show_result(img, result, out_file=img_name)
        for cls, item in enumerate(result):
            if item is None:
                continue
            else:
                for row in item:
                    label = '%s %.2f' % (model.CLASSES[int(cls)], row[4])
                    plot_one_box(row[:4], img, label=label, color=colors[cls], line_thickness=3)
        cv2.imwrite('output/' + img_name, img)

    with open('result.json', 'w') as fp:
        json.dump(list(share_submit_result), fp, indent=4, ensure_ascii=False)

    print('time use: %.3fs' % (time.time() - start))