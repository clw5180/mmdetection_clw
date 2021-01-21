# coding=UTF-8<code>
import json
from xml.etree import ElementTree as ET

from cv2 import cv2 
import math
import random
random.seed(10)
from glob import glob
import sys
import os
import shutil
import numpy as np

#root = '/Users/songchen/Downloads/tianchi/'
root = '/home/user/dataset/'

img_paths = root+ 'tile_round1_train_20201231/train_imgs/'
with open(root+'tile_round1_train_20201231/train_annos.json', 'r') as f:
    image_meta = json.load(f, encoding='etf-8')
each_img_meta = {}
for each_item in image_meta:
    each_img_meta[each_item['name']] = []
for idx, each_item in enumerate(image_meta):
    bbox = each_item['bbox']
    bbox.append(each_item['category'])
    each_img_meta[each_item['name']].append(bbox)

center_chip = False
window_w = 1333
window_h = 800

# window_w = 832
# window_h = 832
#window_w = 1536
#window_h = 920

def chiping_training(number_repeat):
    number = 0
    test_dict = {}
    for idx, each_item in enumerate(image_meta):
        # print(each_item)
        bbox = each_item['bbox']   #每一个json item对应的的bbox
        if (bbox[3] - bbox[1])>=window_h or (bbox[2] - bbox[0])>=window_w:
            #window_w = int(1333*1.5)
            #window_h = int(800*1.5)
            print('large bbox:', each_item['name'])
            #使用continue则被忽略了
            continue

        img = cv2.imread(img_paths + each_item['name'])
        h, w = img.shape[:2]

        for now_repeat in range(number_repeat):
            #计算bbox中心坐标
            center_x, center_y = int(bbox[0] + (bbox[2] - bbox[0]) / 2), int((bbox[3] - bbox[1]) / 2 + bbox[1])
            if center_chip:
                x, y, r, b = center_x - window_w // 2, center_y - window_h // 2, center_x + window_w // 2, center_y + window_h // 2
            else:
            #random chips
                min_w, max_w = min(int((bbox[2] - bbox[0]) / 2), int(window_w-(bbox[2] - bbox[0]) / 2)), max(int((bbox[2] - bbox[0]) / 2), int(window_w-(bbox[2] - bbox[0]) / 2))
                min_h, max_h = min(int((bbox[3] - bbox[1]) / 2), int(window_h-(bbox[3] - bbox[1]) / 2)), max(int((bbox[3] - bbox[1]) / 2), int(window_h-(bbox[3] - bbox[1]) / 2))
                random_w = random.randint(min_w, max_w)
                random_h = random.randint(min_h, max_h)
                x, y, r, b = center_x - random_w, center_y - random_h, center_x + (window_w - random_w), center_y + (window_h - random_h)

            # 避免越过图像边界
            x = max(0, x)
            y = max(0, y)
            r = min(r, w)
            b = min(b, h)

            boxes = each_img_meta[each_item['name']] #这张图像所有的bboxes

            annotations = []
            for e_box in boxes:
                #判断该图的每个bbox是否在 该 切图的包围范围内
                if x < e_box[0] < r and y < e_box[1] < b and x < e_box[2] < r and y < e_box[3] < b:
                    e_box = [int(i) for i in e_box]
                    e_box[0] = math.floor(e_box[0] - x)
                    assert(e_box[0]>=0)
                    e_box[1] = math.floor(e_box[1] - y)
                    assert(e_box[1]>=0)
                    e_box[2] = math.ceil(e_box[2] - x)
                    assert(e_box[2]>=0)
                    e_box[3] = math.ceil(e_box[3] - y)
                    assert(e_box[3]>=0)

                    test_dict[number] = {'name':  each_item['name'][:-4]+'_'+str(idx)+'_chip_'+str(now_repeat)+'.jpg', 'image_height': window_h, 'image_width': window_w, 'category':e_box[4], 'bbox':{'0': e_box[0], '1':e_box[1], '2':e_box[2], '3':e_box[3]}}
                    number = number+1
                    annotations.append(e_box)

            #print('process id:', idx)
            slice_img = img[y:b, x:r]
            if y>b or x>r:
                print('*****',each_item['name'])
                input()
            slice_name= each_item['name'][:-4]+'_'+str(idx)+'_chip_'+str(now_repeat)+'.jpg'
            cv2.imwrite(root+chip_strategy +'/images/' + slice_name, slice_img)

        if idx%100==0:
            print('process id:', idx)

            #检查,注意因为使用了continue，这里必须注释掉
            #num_images = glob(root+chip_strategy+'/images/*')
            #assert(len(num_images)==(idx+1)*number_repeat)

    print('Number of bboxes:', number)
    num_images = glob(root+chip_strategy+'/images/*')
    print('Number of chips: ', num_images)
    print('process id:', idx)
    
    save_json_path = root+chip_strategy+"/Annotations/record_all.json"
    with open(save_json_path,"w") as f:
        json.dump(test_dict,f)


import os
import json
import codecs

def to_voc(rawLabelDir, anno_dir):

    class_name_dic = {
    "0": "背景",
    "1": "边异常",
    "2": "角异常",
    "3": "白色点瑕疵",
    "4": "浅色块瑕疵",
    "5": "深色点块瑕疵",
    "6": "光圈瑕疵"
    }

    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)
    with open(rawLabelDir) as f:
        annos=json.load(f)

    #
    image_ann={}
    for i in range(len(annos)):
        anno=annos[str(i)]
        name = anno['name']
        if name not in image_ann:
            image_ann[name]=[]
        image_ann[name].append(i)
    #
    for name in image_ann.keys():
        indexs=image_ann[name]
        height, width = annos[str(indexs[0])]["image_height"], annos[str(indexs[0])]["image_width"]
        #
        with codecs.open(anno_dir + name[:-4] + '.xml', 'w', 'utf-8') as xml:
            xml.write('<annotation>\n')
            xml.write('\t<filename>' + name + '</filename>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(3) + '</depth>\n')
            xml.write('\t</size>\n')
            cnt = 0
            for inx in indexs:
                obj = annos[str(inx)]
                assert name == obj['name']
                bbox = obj['bbox']
                category = obj['category']
                if isinstance(bbox, dict):
                    xmin, ymin, xmax, ymax = bbox['0'],bbox['1'],bbox['2'],bbox['3']
                else:
                    xmin, ymin, xmax, ymax = bbox

                class_name = class_name_dic[str(category)]
                #
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + class_name + '</name>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                cnt += 1
            assert cnt > 0
            xml.write('</annotation>')


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars



def convert(xml_list, xml_dir, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    list_fp = xml_list
    image_id=1
    # 标注基本结构
    json_dict = {"images":[],
                "type": "instances",
                "annotations": [],
                "categories": []}
    # import mmcv
    # 检测框的ID起始值
    START_BOUNDING_BOX_ID = 1
    # 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
    PRE_DEFINE_CATEGORIES ={
    "边异常":0,
    "角异常":1,
    "白色点瑕疵":2,
    "浅色块瑕疵":3,
    "深色点块瑕疵":4,
    "光圈瑕疵":5
    }
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    lost=0
    ratios=[]
    for line in list_fp:
        line = line.strip()
        print(" Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = root.find('filename').text
        # 取出图片名字
        image_id+=1
        size = get_and_check(root, 'size', 1)
        # 图片的基本信息
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename,
                'height': height,
                'width': width,
                'id':image_id}
        #del image['file_name']
        json_dict['images'].append(image)
        # 处理每个标注的检测框
        for obj in get(root, 'object'):
            # 取出检测框类别名称
            category = get_and_check(obj, 'name', 1).text
            # 更新类别ID字典
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            ratios.append(float(o_width/o_height) if o_width>o_height else o_height/o_width)
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1
    import matplotlib.pyplot as plt
    plt.hist(ratios)
    plt.show()

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    #mmcv.dump(json_dict, json_file)
    #print(type(json_dict))
    #print(json_dict)
    json_data = json.dumps(json_dict)
    with  open(json_file, 'w') as w:
        w.write(json_data)

def voc_coco(root_path, chip_strategy):
    '''
    convert voc to coco format
    '''
    # -*- coding=utf-8 -*-
    #!/usr/bin/python
    import sys
    import os
    import shutil
    import numpy as np
    import json
    import xml.etree.ElementTree as ET
    # import mmcv
    # 检测框的ID起始值
    START_BOUNDING_BOX_ID = 1
    # 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
    PRE_DEFINE_CATEGORIES ={
    "边异常":0,
    "角异常":1,
    "白色点瑕疵":2,
    "浅色块瑕疵":3,
    "深色点块瑕疵":4,
    "光圈瑕疵":5
    }

    if not os.path.exists(os.path.join(root_path,'coco/annotations')):
        os.makedirs(os.path.join(root_path,'coco/annotations'))
    if not os.path.exists(os.path.join(root_path, 'coco/train2017')):
        os.makedirs(os.path.join(root_path, 'coco/train2017'))
    if not os.path.exists(os.path.join(root_path, 'coco/val2017')):
        os.makedirs(os.path.join(root_path, 'coco/val2017'))
    xml_dir = os.path.join(root_path,'voc') #已知的voc的标注

    xml_labels = os.listdir(xml_dir)
    
    train = True
    if train:
        tmp=[]
        lost=0
        for xml in xml_labels:
            img_name = xml[:-4] + '.jpg'
            ishave=os.path.exists(root_path[0:-13]+'/images/'+img_name)
            if not os.path.exists(root_path[0:-13]+'/images/'+img_name):
                lost += 1
                continue
            tmp.append(xml)
        xml_labels=tmp
        print('Lost_chips  {}'.format(lost))

        np.random.shuffle(xml_labels)
        split_point = int(len(xml_labels)/10)

        # split training data and val data
        # validation data
        xml_list = xml_labels[0:split_point]
        json_file = os.path.join(root_path,'coco/annotations/instances_val2017.json')
        convert(xml_list, xml_dir, json_file)
        for xml_file in xml_list:
            img_name = xml_file[:-4] + '.jpg'
            shutil.copy(root_path[0:-13]+'/images/'+img_name,
                        os.path.join(root_path, 'coco/val2017', img_name))
        # train data
        xml_list = xml_labels[split_point:]
        json_file = os.path.join(root_path,'coco/annotations/instances_train2017.json')
        convert(xml_list, xml_dir, json_file)
        for i,xml_file in enumerate(xml_list):
            img_name = xml_file[:-4] + '.jpg'
            # print('{}/{}'.format(i,len(xml_list)))
            shutil.copy(root_path[0:-13]+'/images/'+img_name,
                        os.path.join(root_path, 'coco/train2017', img_name))
    else:
        # np.random.shuffle(xml_labels)
        # validation data
        xml_list = xml_labels
        json_file = os.path.join(root_path,'./instances_test2017.json')
        convert(xml_list, xml_dir, json_file)

chip_strategy = 'chips_strategy_1388_800_version_4'
#chip_strategy = 'chips_strategy_832_832_version_1'
os.makedirs(root+chip_strategy+'/images',exist_ok=True)
os.makedirs(root+chip_strategy+'/Annotations',exist_ok=True)
os.makedirs(root+chip_strategy+'/Annotations/coco',exist_ok=True)
os.makedirs(root+chip_strategy+'/Annotations/voc',exist_ok=True)

number_repeat = 1
chiping_training(number_repeat)

save_json_path = root+chip_strategy+"/Annotations/record_all.json"
to_voc(save_json_path, root+chip_strategy+'/Annotations/voc/')
voc_coco(root+chip_strategy+'/Annotations/', chip_strategy)