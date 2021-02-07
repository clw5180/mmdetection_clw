## clw note: 直接concat模板图, 6 channel 输入, 注意还需要修改第一个卷积层输入通道数,并且frozen_stage=-1,需要重新学习;

from ..builder import PIPELINES
import mmcv
import numpy as np
import os
import cv2

@PIPELINES.register_module
class Concat6(object):
    """Concat two image.

    Args:
        template_path: template images path
    """

    def __init__(self, template_path):
        self.template_path = template_path

    def __call__(self, results):
        #template_name = 'template_' + results['img_info']['filename'].split('_')[0] + '.jpg'
        template_name = results['img_info']['filename'].split('.')[0] + '_t' + '.jpg'
        template_im_name = os.path.join(self.template_path , template_name)
        img_temp = mmcv.imread(template_im_name)
        results['img'] = np.concatenate([results['img'], img_temp], axis=2)
        results['concat'] = True
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(
            self.template_path)
        return repr_str