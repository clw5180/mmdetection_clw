
"""
#  @Time     : 2021/2/5  21:02
#  @Author   : Yufang
#  阿里云比赛在线推理，使用mmd的api
#  @update   : 2/8      21:56 判断img是否有瑕疵
            2/10      15:00 对偶推理
"""


import argparse

import os
from ai_hub import inferServer
import cv2
import torch
import time
import numpy as np
import json
from mmdet.apis import inference_detector, init_detector, dual_inference_detector, dual_inference_detector_v2

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection network inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--img_score_thr',
        type=float,
        default=0.2,
        help='score threshold (default: 0.001)')
    parser.add_argument('--dual_infer', default=True, type=bool, help='whether to dual infer')
    args = parser.parse_args()
    return args


class myserver(inferServer):
    def __init__(self, model, img_score_thr, dual_infer):
        super().__init__(model)
        print("init_myserver")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model#.to(device)
        self.filename = None
        self.img_score_thr = img_score_thr
        self.dual_infer = dual_infer

    def pre_process(self, request):
        print("my_pre_process.")
        # json process
        # file example
        file = request.files['img']
        file_t = request.files['img_t']

        self.filename = file.filename

        file_data = file.read()
        file_data_t = file_t.read()
        data = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR) # 读取来自网络的图片
        data_t = cv2.imdecode(np.frombuffer(file_data_t, np.uint8), cv2.IMREAD_COLOR)  # 读取来自网络的图片

        # height, width, _ = data.shape
        # image_shape = (width, height)

        # print(file.filename)
        # print(image_shape)

        return [data, data_t]

    # predict default run as follow：
    def pridect(self, data):


        start = time.perf_counter()
        data_ori = data[0]
        data_t = data[1]
        if self.dual_infer:
            #result = dual_inference_detector(model, data_ori, data_t)
            result = dual_inference_detector_v2(model, data_ori, data_t)
        else:
            result = inference_detector(model, data_ori)

        elapsed_time = time.perf_counter() - start
        print('The executive time of  model: %.5f' % (elapsed_time))

        return result

    def post_process(self, data):
        # data.cpu()
        output = output2result(data, self.filename, self.img_score_thr)
        # 正常应该经过model预测得到data，执行data.cpu后打包成赛题要求的json返回
        return output#json.dumps(output)


def output2result(result, name, img_score_thr):
    image_name = name
    predict_rslt = []

    #判断图片是否有瑕疵瑕疵
    img_score = 0.0
    for i, res_perclass in enumerate(result):
        for per_class_results in res_perclass:
            _, _, _, _, score = per_class_results
            if score > img_score:
                img_score = score

    print('img score: '+str(img_score))
    if img_score > img_score_thr:
        for i, res_perclass in enumerate(result):
            class_id = i + 1
            for per_class_results in res_perclass:
                xmin, ymin, xmax, ymax, score = per_class_results
                if score < 0.001: #模型通过nms score出来的已经大于一定的值了
                    continue
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                dict_instance = dict()
                dict_instance['name'] = image_name
                dict_instance['category'] = class_id
                dict_instance["score"] = round(float(score), 6)
                dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
                predict_rslt.append(dict_instance)

    return predict_rslt



if __name__ == '__main__':
    args = parse_args()

    model = init_detector(args.config, args.checkpoint)

    myserver = myserver(model,args.img_score_thr, args.dual_infer)
    # run your server, defult ip=localhost port=8080 debuge=false
    myserver.run(debuge=False)  # myserver.run("127.0.0.1", 1234)