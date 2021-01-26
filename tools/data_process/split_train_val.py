# 将tianchi数据标注转为coco格式

import math
import json
import pandas as pd
import shutil
import json
import os
import cv2
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, x2, y2]
    boxes: [boxes_count, (x1, y1, x2, y2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[2], boxes[:, 2])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1+1, 0) * np.maximum(y2 - y1+1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, x2, y2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]+1) * (boxes1[:, 3] - boxes1[:, 1]+1)
    area2 = (boxes2[:, 2] - boxes2[:, 0]+1) * (boxes2[:, 3] - boxes2[:, 1]+1)

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Fabric2COCO:

    def __init__(self,base_save_path,
                 is_mode="train"):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.is_mode = is_mode
        self.patch_size = 1024
        self.stride = int(self.patch_size * 0.8)
        self.root_dir = os.path.join(base_save_path, '{}')
        if not os.path.exists(self.root_dir.format(self.is_mode)):
            os.makedirs(self.root_dir.format(self.is_mode))

    def to_coco(self, anno_file, img_dir):
        aaa = json.load(open(anno_file, 'r'))
        cats = set()
        for item in aaa:
            cats.add(item['category'])

        self._init_categories(cats)
        anno_result = pd.read_json(open(anno_file, "r"))

        name_boxes = {}
        for info in anno_result.values:
            img_name, _, _, cls, box = info
            if img_name not in name_boxes.keys():
                name_boxes[img_name] = {'boxes': [], 'clses': []}
            name_boxes[img_name]['boxes'].append(box)
            name_boxes[img_name]['clses'].append(cls)
        img_name_list_all = list(name_boxes.keys())
        img_name_list_all.sort()
        np.random.seed(0)
        np.random.shuffle(img_name_list_all)
        img_num = len(img_name_list_all)
        if self.is_mode == "train":
            img_name_list = img_name_list_all[:int(img_num * 0.9)]
        elif self.is_mode == "val":
            img_name_list = img_name_list_all[int(img_num * 0.9):]

        for img_name in img_name_list:
            img_prefix = os.path.splitext(img_name)[0]
            boxes = np.array(name_boxes[img_name]['boxes'])
            defect_names = np.array(name_boxes[img_name]['clses'])

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape

            new_img_h = math.ceil(h / self.patch_size) * self.patch_size
            new_img_w = math.ceil(w / self.patch_size) * self.patch_size
            new_img = np.zeros([new_img_h, new_img_w, 3])
            new_img[:h, :w, :] = img

            for i in range(0, new_img_w - self.patch_size, self.stride):
                for j in range(0, new_img_h - self.patch_size, self.stride):
                    img_patch = new_img[j:j + self.patch_size, i:i + self.patch_size, :]
                    iou = compute_overlaps(np.array([[i, j, i + self.patch_size, j + self.patch_size]]), boxes)
                    iou = iou.reshape(-1)
                    index = iou > 0
                    if np.sum(index) == 0:
                        self._clean_img(img_prefix + '-{:0>2}_{:0>2}.jpg'.format(i, j), img_patch)  # 复制文件路径
                        continue
                    boxes_temp = boxes[index]
                    clses_temp = defect_names[index]
                    boxes_temp[:, 0] = boxes_temp[:, 0] - i
                    boxes_temp[:, 1] = boxes_temp[:, 1] - j
                    boxes_temp[:, 2] = boxes_temp[:, 2] - i
                    boxes_temp[:, 3] = boxes_temp[:, 3] - j
                    path_temp = os.path.join(img_dir)
                    self.images.append(
                        self._image(img_prefix + '-{:0>6}_{:0>2}_{:0>2}.jpg'.format(self.img_id, i, j), 1024, 1024))

                    self._cp_img(img_prefix + '-{:0>6}_{:0>2}_{:0>2}.jpg'.format(self.img_id, i, j),
                                 img_patch)  # 复制文件路径

                    if self.img_id % 400 is 0:
                        print("处理到第{}张图片".format(self.img_id))
                    for bbox, label in zip(boxes_temp, clses_temp):
                        annotation = self._annotation(label, bbox)
                        self.annotations.append(annotation)
                        self.ann_id += 1
                    self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['Yuan']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self, cats):
        # 1，2，3，4，5，6个类别
        for idx, cat in enumerate(cats):
            print(cat)
            category = {}
            category['id'] = idx+1
            category['name'] = str(cat)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)  # 返回path最后的文件名
        return image

    def _annotation(self, label, bbox):
        bbox[0] = min(max(bbox[0], 0), self.patch_size - 1)
        bbox[1] = min(max(bbox[1], 0), self.patch_size - 1)
        bbox[2] = min(max(bbox[2], 0), self.patch_size - 1)
        bbox[3] = min(max(bbox[3], 0), self.patch_size - 1)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = []  # np.asarray(points).flatten().tolist()
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation["ignore"] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_name, img_patch):
        cv2.imwrite(os.path.join(self.root_dir.format(self.is_mode), img_name), img_patch)

    #         shutil.copy(img_path, os.path.join(self.root_dir.format(self.is_mode), os.path.basename(img_path)))
    def _clean_img(self, img_name, img_patch):
        if not os.path.exists(os.path.join(self.root_dir.format(self.is_mode + '_clean'))):
            os.makedirs(os.path.join(self.root_dir.format(self.is_mode + '_clean')))
        cv2.imwrite(os.path.join(self.root_dir.format(self.is_mode + '_clean'), img_name), img_patch)

    #         shutil.copy(img_path, os.path.join(self.root_dir.format(self.is_mode), os.path.basename(img_path)))
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '),
                      cls=NpEncoder)  # 缩进设置为1，元素之间用逗号隔开 ， key和内容之间 用冒号隔开

    def split_test(self, test_dir='../dataset/coco/tile_round1_testA_20201231/testA_imgs/'):
        img_name_list = os.listdir(test_dir)
        img_name_list.sort()
        for img_name in tqdm(img_name_list):
            img_prefix = os.path.splitext(img_name)[0]

            img_path = os.path.join(test_dir, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape

            new_img_h = math.ceil(h / self.patch_size) * self.patch_size
            new_img_w = math.ceil(w / self.patch_size) * self.patch_size
            new_img = np.zeros([new_img_h, new_img_w, 3])
            new_img[:h, :w, :] = img

            for i in range(0, new_img_w - self.patch_size, self.stride):
                for j in range(0, new_img_h - self.patch_size, self.stride):
                    img_patch = new_img[j:j + self.patch_size, i:i + self.patch_size, :]

                    self._clean_img(img_prefix + '-{:0>2}_{:0>2}.jpg'.format(i, j), img_patch)  # 复制文件路径


if __name__ == "__main__":


    '''转换有瑕疵的样本为coco格式'''
    # 训练集,划分90%做为训练集，处理需要50分钟
    img_dir = "/home/user/dataset/tile_round1_train_20201231/train_imgs"
    anno_dir = "/home/user/dataset/tile_round1_train_20201231/train_annos.json"
    base_save_path = "/home/user/dataset/2021tianchi/crop"
    fabric2coco = Fabric2COCO(base_save_path)
    train_instance = fabric2coco.to_coco(anno_dir, img_dir)
    base_save_annotation_path = os.path.join(base_save_path, "annotations")
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    fabric2coco.save_coco_json(train_instance, os.path.join(base_save_path , 'instances_{}.json'.format("train")))

    '''转换有瑕疵的样本为coco格式'''
    # 验证集，划分10%做为验证集
    fabric2coco = Fabric2COCO(base_save_path, is_mode="val" )
    train_instance = fabric2coco.to_coco(anno_dir, img_dir)
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    fabric2coco.save_coco_json(train_instance, os.path.join(base_save_path , 'instances_{}.json'.format("val")))

    ''' inference split '''
    # test_dir = '../dataset/coco/tile_round1_testA_20201231/testA_imgs/'
    # fabric2coco = Fabric2COCO(is_mode="test")
    # fabric2coco.split_test(test_dir)