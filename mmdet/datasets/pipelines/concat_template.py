## yf note: 直接concat模板图, online infer 需给定img_t;

from ..builder import PIPELINES
import mmcv
import numpy as np
import os

@PIPELINES.register_module()
class LoadTemplate(object):
    """Load template image.

    Args:
        template_path: template images path
    """

    def __init__(self, template_path):
        self.template_path = template_path

    def __call__(self, results):
        if 'img_t' not in results or results['img_t'] is None:
            template_name = results['img_info']['filename'].split('.')[0] + '_t' + '.jpg'
            template_im_name = os.path.join(self.template_path , template_name)
            img_t = mmcv.imread(template_im_name)
            results['img_t'] = img_t
            results['concat'] = True
            results['img_fields'] = ['img','img_t']
        else:
            results['concat'] = True
            results['img_fields'] = ['img', 'img_t']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(
            self.template_path)
        return repr_str


@PIPELINES.register_module()
class ConcatTemplate(object):
    """Concat two image.
    """


    def __call__(self, results):

        results['img'] = np.concatenate([results['img'], results['img_t']], axis=2)

        return results
