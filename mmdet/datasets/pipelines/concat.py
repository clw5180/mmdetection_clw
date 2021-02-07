## clw note: 组合三通道, 包括原图,模板图和差分图, 3 channel 输入, 因此要转为黑白

from ..builder import PIPELINES
import mmcv
import numpy as np
import os
import cv2

@PIPELINES.register_module
class Concat(object):
    """Concat two image.

    Args:
        template_path: template images path
    """

    def __init__(self, template_path):
        self.template_path = template_path

    def __call__(self, results):
        #template_name = 'template_' + results['img_info']['filename'].split('_')[0] + '.jpg'
        #print('aaaaaa')
        template_name = results['img_info']['filename'].split('.')[0] + '_t' + '.jpg'
        template_im_name = os.path.join(self.template_path , template_name)
        img_temp = cv2.imread(template_im_name)
        img_ori = cv2.cvtColor(results['img'], cv2.COLOR_BGR2GRAY)
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img_diff = cv2.subtract(img_ori, img_temp)
        # img_diff *= 3
        results['img'] = np.concatenate([np.expand_dims(img_ori, axis=2), np.expand_dims(img_diff, axis=2),
                                         np.expand_dims(img_temp, axis=2)], axis=2)
        #print(results['img'].shape)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(
            self.template_path)
        return repr_str