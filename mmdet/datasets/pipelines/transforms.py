import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES
import cv2

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
                # 修改 WH对应乘
                # h, w = results['img'].shape[:2]
                # scale_factor = results['scale_factor']
                # results['scale'] = tuple([int(w * scale_factor[0]),int(h * scale_factor[0])])
                # yf add 防止模板图和瑕疵图shape不一致，用比例不能cat
                # results['scale'] = results['scale_factor']
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            # yf add 防止模板图和瑕疵图shape不一致，用比例不能cat
            # else:
                # results.pop('scale')
                # if 'scale_factor' in results:
                #     results.pop('scale_factor')
                # self._random_scale(results)



        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        # yf add 防止模板图和瑕疵图shape不一致，用比例不能cat
        # results['scale_factor'] = results['scale']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image wil
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image wil
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
         of 0.3, vertically with probability of 0.5

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True):
        if crop_type not in [
            'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                    bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
        backend (str): Image rescale backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self, scale_factor=1, backend='cv2'):
        self.scale_factor = scale_factor
        self.backend = backend

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """

        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key],
                    self.scale_factor,
                    interpolation='nearest',
                    backend=self.backend)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        """Call function to expand images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_img = np.empty((int(h * ratio), int(w * ratio), c),
                                  img.dtype)
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full((int(h * ratio), int(w * ratio), c),
                                 self.mean,
                                 dtype=img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img

        results['img'] = expand_img
        # expand bboxes
        for key in results.get('bbox_fields', []):
            results[key] = results[key] + np.tile(
                (left, top), 2).astype(results[key].dtype)

        # expand masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].expand(
                int(h * ratio), int(w * ratio), top, left)

        # expand segs
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results[key] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str


@PIPELINES.register_module()
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        if self.bbox_clip_border:
                            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                        # mask fields
                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                   patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class Corrupt(object):
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.

    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str


@PIPELINES.register_module()
class Albu(object):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            if albumentations.__version__ < '0.5':
                results['masks'] = results['masks'].masks
            else:
                results['masks'] = [mask for mask in results['masks'].masks]

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], results['image'].shape[0],
                        results['image'].shape[1])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class RandomCenterCropPad(object):
    """Random center crop and random around padding for CornerNet.

    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from :class:`RandomCrop`, the output
    shape may not equal to ``crop_size`` strictly. We choose a random value
    from ``ratios`` and the output shape could be larger or smaller than
    ``crop_size``. The padding operation is also different from :class:`Pad`,
    here we use around padding instead of right-bottom padding.

    The relation between output image (padding image) and original image:

    .. code:: text

                        output image

               +----------------------------+
               |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
               |          padded area       |
               +----------------------------+

    There are 5 main areas in the figure:

    - output image: output image of this operation, also called padding
      image in following instruction.
    - original image: input image of this operation.
    - padded area: non-intersect area of output image and original image.
    - cropped area: the overlap of output image and original image.
    - center range: a smaller area where random center chosen from.
      center range is computed by ``border`` and original image's shape
      to avoid our random center is too close to original image's border.

    Also this operation act differently in train and test mode, the summary
    pipeline is listed below.

    Train pipeline:

    1. Choose a ``random_ratio`` from ``ratios``, the shape of padding image
       will be ``random_ratio * crop_size``.
    2. Choose a ``random_center`` in center range.
    3. Generate padding image with center matches the ``random_center``.
    4. Initialize the padding image with pixel value equals to ``mean``.
    5. Copy the cropped area to padding image.
    6. Refine annotations.

    Test pipeline:

    1. Compute output shape according to ``test_pad_mode``.
    2. Generate padding image with center matches the original image
       center.
    3. Initialize the padding image with pixel value equals to ``mean``.
    4. Copy the ``cropped area`` to padding image.

    Args:
        crop_size (tuple | None): expected size after crop, final size will
            computed according to ratio. Requires (h, w) in train mode, and
            None in test mode.
        ratios (tuple): random select a ratio from tuple and crop image to
            (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode.
        border (int): max distance from center select area to image border.
            Only available in train mode.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
        test_mode (bool): whether involve random variables in transform.
            In train mode, crop_size is fixed, center coords and ratio is
            random selected from predefined lists. In test mode, crop_size
            is image's original shape, center coords and ratio is fixed.
        test_pad_mode (tuple): padding method and padding shape value, only
            available in test mode. Default is using 'logical_or' with
            127 as padding shape value.

            - 'logical_or': final_shape = input_shape | padding_shape_value
            - 'size_divisor': final_shape = int(
              ceil(input_shape / padding_shape_value) * padding_shape_value)
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    """

    def __init__(self,
                 crop_size=None,
                 ratios=(0.9, 1.0, 1.1),
                 border=128,
                 mean=None,
                 std=None,
                 to_rgb=None,
                 test_mode=False,
                 test_pad_mode=('logical_or', 127),
                 bbox_clip_border=True):
        if test_mode:
            assert crop_size is None, 'crop_size must be None in test mode'
            assert ratios is None, 'ratios must be None in test mode'
            assert border is None, 'border must be None in test mode'
            assert isinstance(test_pad_mode, (list, tuple))
            assert test_pad_mode[0] in ['logical_or', 'size_divisor']
        else:
            assert isinstance(crop_size, (list, tuple))
            assert crop_size[0] > 0 and crop_size[1] > 0, (
                'crop_size must > 0 in train mode')
            assert isinstance(ratios, (list, tuple))
            assert test_pad_mode is None, (
                'test_pad_mode must be None in train mode')

        self.crop_size = crop_size
        self.ratios = ratios
        self.border = border
        # We do not set default value to mean, std and to_rgb because these
        # hyper-parameters are easy to forget but could affect the performance.
        # Please use the same setting as Normalize for performance assurance.
        assert mean is not None and std is not None and to_rgb is not None
        self.to_rgb = to_rgb
        self.input_mean = mean
        self.input_std = std
        if to_rgb:
            self.mean = mean[::-1]
            self.std = std[::-1]
        else:
            self.mean = mean
            self.std = std
        self.test_mode = test_mode
        self.test_pad_mode = test_pad_mode
        self.bbox_clip_border = bbox_clip_border

    def _get_border(self, border, size):
        """Get final border for the target size.

        This function generates a ``final_border`` according to image's shape.
        The area between ``final_border`` and ``size - final_border`` is the
        ``center range``. We randomly choose center from the ``center range``
        to avoid our random center is too close to original image's border.
        Also ``center range`` should be larger than 0.

        Args:
            border (int): The initial border, default is 128.
            size (int): The width or height of original image.
        Returns:
            int: The final border.
        """
        k = 2 * border / size
        i = pow(2, np.ceil(np.log2(np.ceil(k))) + (k == int(k)))
        return border // i

    def _filter_boxes(self, patch, boxes):
        """Check whether the center of each box is in the patch.

        Args:
            patch (list[int]): The cropped area, [left, top, right, bottom].
            boxes (numpy array, (N x 4)): Ground truth boxes.

        Returns:
            mask (numpy array, (N,)): Each box is inside or outside the patch.
        """
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (
                center[:, 0] < patch[2]) * (
                       center[:, 1] < patch[3])
        return mask

    def _crop_image_and_paste(self, image, center, size):
        """Crop image with a given center and size, then paste the cropped
        image to a blank image with two centers align.

        This function is equivalent to generating a blank image with ``size``
        as its shape. Then cover it on the original image with two centers (
        the center of blank image and the random center of original image)
        aligned. The overlap area is paste from the original image and the
        outside area is filled with ``mean pixel``.

        Args:
            image (np array, H x W x C): Original image.
            center (list[int]): Target crop center coord.
            size (list[int]): Target crop size. [target_h, target_w]

        Returns:
            cropped_img (np array, target_h x target_w x C): Cropped image.
            border (np array, 4): The distance of four border of
                ``cropped_img`` to the original image area, [top, bottom,
                left, right]
            patch (list[int]): The cropped area, [left, top, right, bottom].
        """
        center_y, center_x = center
        target_h, target_w = size
        img_h, img_w, img_c = image.shape

        x0 = max(0, center_x - target_w // 2)
        x1 = min(center_x + target_w // 2, img_w)
        y0 = max(0, center_y - target_h // 2)
        y1 = min(center_y + target_h // 2, img_h)
        patch = np.array((int(x0), int(y0), int(x1), int(y1)))

        left, right = center_x - x0, x1 - center_x
        top, bottom = center_y - y0, y1 - center_y

        cropped_center_y, cropped_center_x = target_h // 2, target_w // 2
        cropped_img = np.zeros((target_h, target_w, img_c), dtype=image.dtype)
        for i in range(img_c):
            cropped_img[:, :, i] += self.mean[i]
        y_slice = slice(cropped_center_y - top, cropped_center_y + bottom)
        x_slice = slice(cropped_center_x - left, cropped_center_x + right)
        cropped_img[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_center_y - top, cropped_center_y + bottom,
            cropped_center_x - left, cropped_center_x + right
        ],
            dtype=np.float32)

        return cropped_img, border, patch

    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        boxes = results['gt_bboxes']
        while True:
            scale = random.choice(self.ratios)
            new_h = int(self.crop_size[0] * scale)
            new_w = int(self.crop_size[1] * scale)
            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = random.randint(low=w_border, high=w - w_border)
                center_y = random.randint(low=h_border, high=h - h_border)

                cropped_img, border, patch = self._crop_image_and_paste(
                    img, [center_y, center_x], [new_h, new_w])

                mask = self._filter_boxes(patch, boxes)
                # if image do not have valid bbox, any crop patch is valid.
                if not mask.any() and len(boxes) > 0:
                    continue

                results['img'] = cropped_img
                results['img_shape'] = cropped_img.shape
                results['pad_shape'] = cropped_img.shape

                x0, y0, x1, y1 = patch

                left_w, top_h = center_x - x0, center_y - y0
                cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

                # crop bboxes accordingly and clip to the image boundary
                for key in results.get('bbox_fields', []):
                    mask = self._filter_boxes(patch, results[key])
                    bboxes = results[key][mask]
                    bboxes[:, 0:4:2] += cropped_center_x - left_w - x0
                    bboxes[:, 1:4:2] += cropped_center_y - top_h - y0
                    if self.bbox_clip_border:
                        bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
                        bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
                    keep = (bboxes[:, 2] > bboxes[:, 0]) & (
                            bboxes[:, 3] > bboxes[:, 1])
                    bboxes = bboxes[keep]
                    results[key] = bboxes
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            labels = results['gt_labels'][mask]
                            labels = labels[keep]
                            results['gt_labels'] = labels
                        if 'gt_masks' in results:
                            raise NotImplementedError(
                                'RandomCenterCropPad only supports bbox.')

                # crop semantic seg
                for key in results.get('seg_fields', []):
                    raise NotImplementedError(
                        'RandomCenterCropPad only supports bbox.')
                return results

    def _test_aug(self, results):
        """Around padding the original image without cropping.

        The padding mode and value are from ``test_pad_mode``.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        results['img_shape'] = img.shape
        if self.test_pad_mode[0] in ['logical_or']:
            target_h = h | self.test_pad_mode[1]
            target_w = w | self.test_pad_mode[1]
        elif self.test_pad_mode[0] in ['size_divisor']:
            divisor = self.test_pad_mode[1]
            target_h = int(np.ceil(h / divisor)) * divisor
            target_w = int(np.ceil(w / divisor)) * divisor
        else:
            raise NotImplementedError(
                'RandomCenterCropPad only support two testing pad mode:'
                'logical-or and size_divisor.')

        cropped_img, border, _ = self._crop_image_and_paste(
            img, [h // 2, w // 2], [target_h, target_w])
        results['img'] = cropped_img
        results['pad_shape'] = cropped_img.shape
        results['border'] = border
        return results

    def __call__(self, results):
        img = results['img']
        assert img.dtype == np.float32, (
            'RandomCenterCropPad needs the input image of dtype np.float32,'
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline')
        h, w, c = img.shape
        assert c == len(self.mean)
        if self.test_mode:
            return self._test_aug(results)
        else:
            return self._train_aug(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'ratios={self.ratios}, '
        repr_str += f'border={self.border}, '
        repr_str += f'mean={self.input_mean}, '
        repr_str += f'std={self.input_std}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        repr_str += f'test_mode={self.test_mode}, '
        repr_str += f'test_pad_mode={self.test_pad_mode}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class CutOut(object):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0)):

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, results):
        """Call function to drop some regions of image."""
        results['cutout_overlaps'] = np.zeros(results['gt_bboxes'].shape[0])
        h, w, c = results['img'].shape
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:  # clw note: False
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            ##### clw note: 最好先计算所有 results['gt_bboxes'] 和 cutout的iou,比如cutout完全遮挡了小目标,那就应该取消cutout操作;
            gt_bboxes = results['gt_bboxes']
            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            cutout_patch = np.array((x1, y1, x2, y2), dtype=np.float32)
            cutout_patch = cutout_patch.reshape(1, 4)
            overlaps = self._compute_overlap( cutout_patch, gt_bboxes )  # 比如某些小目标被完全覆盖了,iou就等于1,这时要完全跳过
            if overlaps.sum() == 0:
                continue

            overlaps = overlaps[0]  # 2 dim arrary -> 1 dim
            for i, overlap in enumerate(overlaps):
                if overlap == 0:  # cutout在背景区域
                    continue
                elif overlap > 0 and (gt_bboxes[i][3] - gt_bboxes[i][1]) * (gt_bboxes[i][2] - gt_bboxes[i][0]) < cutout_w * cutout_h : # 不让cutout到小目标,碰一点都不行
                    continue
                elif overlap < 0.5:
                    if results['cutout_overlaps'][i] + overlap > 0.5:
                        continue
                    else:
                        results['cutout_overlaps'][i] += overlap
                        results['img'][y1:y2, x1:x2, :] = self.fill_in
                else:
                    continue

        return results

    def _compute_overlap(self, a, b):  # clw added
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlaps = intersection / ua
        #index = overlap > over_threshold
        return overlaps

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'
        return repr_str


@PIPELINES.register_module()
class GtBoxBasedCrop(object):
    """
           Crop around the gt bbox.
           Note:
               Here 'img_shape' is change to the shape of img_cropped.
       """

    def __init__(self, crop_size):
        self.crop_size = crop_size  # (w, h)

    def __call__(self, results):

        img = results['img']
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']

        img_cropped, gt_bboxes_cropped, gt_labels_cropped = \
            self._crop_patch(img, gt_bboxes, gt_labels)

        results['img'] = img_cropped
        results['gt_bboxes'] = gt_bboxes_cropped
        results['gt_labels'] = gt_labels_cropped
        results['img_shape'] = img_cropped.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(crop_size={})'.format(self.crop_size)
        return repr_str

    def _crop_patch(self, img, gt_bboxes, gt_labels):

        H, W, C = img.shape
        px, py = self.crop_size

        if px > W or py > H:
            pad_w, pad_h = 0, 0

            if px > W:
                pad_w = px - W
                W = px
            if py > H:
                pad_h = py - H
                H = py

            img = cv2.copyMakeBorder(img, 0, int(pad_h), 0, int(pad_w), cv2.BORDER_CONSTANT, 0)

        obj_num = gt_bboxes.shape[0]
        select_gt_id = np.random.randint(0, obj_num)
        x1, y1, x2, y2 = gt_bboxes[select_gt_id, :]

        if x2 - x1 > px:
            nx = np.random.randint(x1, x2 - px + 1)
        else:
            nx = np.random.randint(max(x2 - px, 0), min(x1 + 1, W - px + 1))

        if y2 - y1 > py:
            ny = np.random.randint(y1, y2 - py + 1)
        else:
            ny = np.random.randint(max(y2 - py, 0), min(y1 + 1, H - py + 1))

        patch_coord = np.zeros((1, 4), dtype="int")
        patch_coord[0, 0] = nx
        patch_coord[0, 1] = ny
        patch_coord[0, 2] = nx + px
        patch_coord[0, 3] = ny + py

        index = self._compute_overlap(patch_coord, gt_bboxes, 0.5)
        index = np.squeeze(index, axis=0)
        index[select_gt_id] = True

        patch = img[ny: ny + py, nx: nx + px, :]
        gt_bboxes = gt_bboxes[index, :]
        gt_labels = gt_labels[index]

        gt_bboxes[:, 0] = np.maximum(gt_bboxes[:, 0] - patch_coord[0, 0], 0)
        gt_bboxes[:, 1] = np.maximum(gt_bboxes[:, 1] - patch_coord[0, 1], 0)
        gt_bboxes[:, 2] = np.minimum(gt_bboxes[:, 2] - patch_coord[0, 0], px - 1)
        gt_bboxes[:, 3] = np.minimum(gt_bboxes[:, 3] - patch_coord[0, 1], py - 1)

        return patch, gt_bboxes, gt_labels

    def _compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index


@PIPELINES.register_module()
class RandomShiftGtBBox(object):
    def __init__(self, shift_rate=0.2, iou_threshold=0.05, overlap_threshold=0.1):
        """
            Randomly shift the gt_bboxes.
            - inputs:
                shift_rate: the probability to shift a gt bbox.
                iou_threshold: the IoU thredshold to keep the gt_bbox.
                overlap_thred: the overlap thredshold to keep the gt_bbox.
        """

        self.shift_rate = shift_rate
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold

    def __call__(self, results):

        shift_flag = np.random.rand(results['gt_bboxes'].shape[0]) < self.shift_rate

        shift_id = np.where(shift_flag)[0]
        keep_id = np.where(~shift_flag)[0]

        if len(shift_id) == 0:  # no shifting
            return results

        shift_gt_bboxes = results['gt_bboxes'][shift_id, :]
        keep_gt_bboxes = results['gt_bboxes'][keep_id, :]
        shift_gt_labels = results['gt_labels'][shift_id]
        keep_gt_labels = results['gt_labels'][keep_id]

        src_img = results['img']

        gt_bboxes = keep_gt_bboxes
        gt_labels = keep_gt_labels
        shift_gt_bbox_list = []

        mask = np.zeros((src_img.shape[0], src_img.shape[1], 1), np.uint8)
        for i in range(len(shift_gt_bboxes)):  # inpainting mask
            ltx = int(shift_gt_bboxes[i, 0])
            lty = int(shift_gt_bboxes[i, 1])
            brx = int(shift_gt_bboxes[i, 2])
            bry = int(shift_gt_bboxes[i, 3])

            mask[lty: bry, ltx: brx, :] = 1

            shift_gt_bbox_list.append(
                src_img[lty: bry, ltx: brx, :].copy()
            )

        src_img = cv2.inpaint(src_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

        h = np.min([src_img.shape[0], results['ori_shape'][0]])
        w = np.min([src_img.shape[1], results['ori_shape'][1]])

        for i in range(len(shift_gt_bbox_list)):
            copy_image = shift_gt_bbox_list[i]
            copy_image_h, copy_image_w = copy_image.shape[:2]
            for attempt in range(10):
                if copy_image_h >= h or copy_image_w >= w:
                    break
                top = np.random.randint(0, h - copy_image_h)
                left = np.random.randint(0, w - copy_image_w)
                box1 = np.array([left, top, left + copy_image_w, top + copy_image_h])
                check_overlap_flag = self._check_box_overlap(box1, gt_bboxes)
                if check_overlap_flag:
                    src_img[top:top + copy_image_h, left:left + copy_image_w, :] = copy_image
                    gt_bboxes = np.concatenate((gt_bboxes, box1[None, :]), axis=0)
                    gt_labels = np.concatenate((gt_labels, shift_gt_labels[i].reshape(1)), axis=0)

                    break

        results['img'] = src_img
        results['gt_bboxes'] = gt_bboxes.astype(results['gt_bboxes'].dtype)
        results['gt_labels'] = gt_labels

        return results

    def _check_box_overlap(self, box1, box_list):
        len_box_list = len(box_list)

        if len_box_list == 0:
            return True

        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b_x1_list, b_y1_list, b_x2_list, b_y2_list = box_list[:, 0], box_list[:, 1], box_list[:, 2], box_list[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1, b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2, b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2, b_y2_list)
        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        box_area_list = np.maximum(b_x2_list - b_x1_list, 0) * np.maximum(b_y2_list - b_y1_list, 0)
        iou_list = inter_area / (box1_area + box_area_list - inter_area)

        box1_overlap = inter_area / box1_area
        box_list_overlap = inter_area / box_area_list

        if np.max(iou_list) > self.iou_threshold:
            return False
        if np.max(box1_overlap) > self.overlap_threshold:
            return False
        if np.max(box_list_overlap) > self.overlap_threshold:
            return False

        return True


@PIPELINES.register_module()
class RandomVerticalFlip(object):
    """Flip the image & bbox & mask vertically.

    If the input dict contains the key "vertical_flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes vertically.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        h = img_shape[0]
        flipped = bboxes.copy()
        flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
        flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        return flipped

    def __call__(self, results):
        if 'vertical_flip' not in results or results['vertical_flip'] == None:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['vertical_flip'] = flip
        if results['vertical_flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction='vertical')
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[::-1, :] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)

@PIPELINES.register_module()
class BboxesJitter(object):
    def __init__(self, shift_ratio=0.1):
        self.shift_ratio = shift_ratio

    def __call__(self, results):
        img, gt_bboxes = results['img'], results['gt_bboxes']
        h, w, c = img.shape

        # box_w,box_h
        box_w, box_h = gt_bboxes[:, 2] - gt_bboxes[:, 0], gt_bboxes[:, 3] - gt_bboxes[:, 1]

        left = np.random.uniform(box_w * (-self.shift_ratio), box_w * self.shift_ratio)
        top = np.random.uniform(box_h * (-self.shift_ratio), box_h * self.shift_ratio)
        right = np.random.uniform(box_w * (-self.shift_ratio), box_w * self.shift_ratio)
        buttom = np.random.uniform(box_h * (-self.shift_ratio), box_h * self.shift_ratio)

        gt_bboxes[:, 0] = np.maximum(0, gt_bboxes[:, 0] + left).astype(int)
        gt_bboxes[:, 1] = np.maximum(0, gt_bboxes[:, 1] + top).astype(int)
        gt_bboxes[:, 2] = np.minimum(w - 1, gt_bboxes[:, 2] + right).astype(int)
        gt_bboxes[:, 3] = np.minimum(h - 1, gt_bboxes[:, 3] + buttom).astype(int)
        results['gt_bboxes'] = gt_bboxes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(shift_ratio={})".format(self.shift_ratio)
        return repr_str


import os
import json
@PIPELINES.register_module()
class Mixup(object):   # clw note: refer to https://github.com/Wakinguup/Underwater_detection/blob/c3d971046cb7d8f6ec8213e9d9fc29c5f28f8412/code/mmdet/datasets/pipelines/transforms.py
    """Mixup images & bbox
    Args:
        prob (float): the probability of carrying out mixup process.
        lambd (float): the parameter for mixup.
        json_path (string): the path to dataset json file.
    """

    def __init__(self, prob=0.5,
                 json_path='data/coco/annotations/pgtrainval2017.json',
                 img_path='data/coco/images/'):
        #self.lambd = np.random.beta(1.5, 1.5)  # clw note: the paper use this value, but the loss need to be weighted !!!
        self.lambd = 0.5
        self.prob = prob
        self.json_path = json_path
        self.img_path = img_path
        with open(json_path, 'r') as json_file:
            all_labels = json.load(json_file)
        self.all_labels = all_labels

    def get_img2(self, img1):
        # random get image2 for mixup
        idx2 = np.random.choice(np.arange(len(self.all_labels['images'])))  # clw note: except img1, TODO
        img2_fn = self.all_labels['images'][idx2]['file_name']
        img2_id = self.all_labels['images'][idx2]['id']
        img2_path = os.path.join(self.img_path , img2_fn)
        img2 = cv2.imread(img2_path)

        ### clw modify: resize img2 and boxes2, as img1, so the size is same;
        scale_w = img1.shape[1] / img2.shape[1]
        scale_h = img1.shape[0] / img2.shape[0]
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        ######################

        # get image2 label
        labels2 = []
        boxes2 = []
        for annt in self.all_labels['annotations']:  # clw note: can call coco api to get the corresponding ann
            if annt['image_id'] == img2_id:
                labels2.append(np.int64(annt['category_id']))
                boxes2.append([np.float32(annt['bbox'][0] * scale_w),
                               np.float32(annt['bbox'][1] * scale_h),
                               np.float32((annt['bbox'][0] + annt['bbox'][2]) * scale_w),
                               np.float32((annt['bbox'][1] + annt['bbox'][3]) * scale_h)])
                               # np.float32((annt['bbox'][0] + annt['bbox'][2] - 1) * scale_w),  # clw note: -1 or not,  TODO
                               # np.float32((annt['bbox'][1] + annt['bbox'][3] - 1) * scale_h)])
        return img2, labels2, boxes2, img2_fn

    def __call__(self, results):
        if random.uniform(0, 1) < self.prob:
            img1 = results['img']
            labels1 = results['gt_labels']

            img2, labels2, boxes2, img2_fn = self.get_img2(img1)
            # if labels2 != []:
            #     break


            ############################################
            assert img1.shape[0] == img2.shape[0]
            assert img1.shape[1] == img2.shape[1]

            height = img1.shape[0]
            width = img1.shape[1]

            if labels2 == []:  # clw note: if sample 2 is pure negative sample
                self.lambd = 0.9  # float(round(random.uniform(0.5,0.9),1))
                # mix image
                # mixup_image = np.zeros([height, width, 3], dtype='float32')
                # mixup_image[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * self.lambd
                # mixup_image[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - self.lambd)
                # mixup_image = mixup_image.astype('uint8')
                mixup_image = img1 * self.lambd + img2 * (1. - self.lambd)
            else:
                self.lambd = 0.5
                # mix image
                mixup_image = img1 * self.lambd + img2 * (1. - self.lambd)
                #mixup_image = mixup_image.astype('uint8')

                # mix labels
                results['gt_labels'] = np.hstack((labels1, np.array(labels2)))
                results['gt_bboxes'] = np.vstack((list(results['gt_bboxes']), boxes2))

                ### 可视化确认结果无误
                filename = results['img_info']['file_name']
                import cv2
                img_out = mixup_image.copy()
                for box in results['gt_bboxes']:
                    cv2.rectangle(img_out, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=3,
                                  lineType=cv2.LINE_AA)

                cv2.imwrite('/home/user/' + filename, img1)
                cv2.imwrite('/home/user/' + img2_fn, img2)
                cv2.imwrite('/home/user/' + filename.split('.')[0] + '_' + img2_fn.split('.')[0] + '.jpg', img_out)
                ###

            results['img'] = mixup_image



            # if the image2 has not bboxes, the 'gt_labels' and 'gt_bboxes' need to be doubled
            # so at the end the half of loss weight can be added as 1 instead of 0.5
            # if boxes2 == []:
            #     results['gt_labels'] = np.hstack((labels1, labels1))
            #     results['gt_bboxes'] = np.vstack((list(results['gt_bboxes']), list(results['gt_bboxes'])))
            # else:

            return results


    def __repr__(self):
        return self.__class__.__name__ + '(prob={}, lambd={}, mixup={}, json_path={}, img_path={})'.format(self.prob,
                                                                                                           self.lambd,
                                                                                                           self.mixup,
                                                                                                           self.json_path,
                                                                                                           self.img_path)



@PIPELINES.register_module
class ReplaceBackground(object):
    '''
    为更好抑制假阳，在训练时引⼊阴性样本数据，具体实现为：训练时，以⼀定的概率丢弃当前随机抽取的阳性Roi，
    从阴性样本数据中随机抽取⼀张阴性Roi作为背景，把当前的阳性样本中的gt_box贴到阴性背景中作为训练样本，
    ⽤cv2.inpaint对贴合的gt_box边缘进⾏修复。
    '''
    def __init__(self, prob=0.2,
                 json_path='data/coco/annotations/pgtrainval2017.json',
                 img_path='data/coco/images/'):

        self.prob = prob
        self.json_path = json_path
        self.img_path = img_path
        with open(json_path, 'r') as json_file:
            all_labels = json.load(json_file)
        self.all_labels = all_labels

    def get_img2(self, img1):
        # random get image2 for mixup
        idx2 = np.random.choice(np.arange(len(self.all_labels['images'])))  # clw note: except img1, TODO
        img2_fn = self.all_labels['images'][idx2]['file_name']
        img2_id = self.all_labels['images'][idx2]['id']
        img2_path = os.path.join(self.img_path , img2_fn)
        img2 = cv2.imread(img2_path)

        ### clw modify: resize img2 and boxes2, as img1, so the size is same;
        scale_w = img1.shape[1] / img2.shape[1]
        scale_h = img1.shape[0] / img2.shape[0]
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        ######################
        return img2, img2_fn


    def __call__(self, results):
        if random.uniform(0, 1) < self.prob:
            src_img = results['img']
            # labels1 = results['gt_labels']

            bg_img, bg_img_fn = self.get_img2(src_img)

            ############################################
            assert src_img.shape[0] == bg_img.shape[0]
            assert src_img.shape[1] == bg_img.shape[1]

            # mask = np.zeros((bg_img.shape[0], bg_img.shape[1], 1), np.uint8)              # mask = cv2.cvtColor(bg_img, cv2.COLOR_RGB2GRAY)

            tmp_gt_bboxes = []
            tmp_gt_labels = []
            for i in range(len(results['gt_bboxes'])):
                x1 = int(results['gt_bboxes'][i][0])
                y1 = int(results['gt_bboxes'][i][1])
                x2 = int(results['gt_bboxes'][i][2])
                y2 = int(results['gt_bboxes'][i][3])
                # 太小的不适合做泊松融合, 会报bug,
                if x2-x1 <= 3 or y2-y1 <= 3:
                    continue
                else:
                    tmp_gt_bboxes.append( results['gt_bboxes'][i] )
                    tmp_gt_labels.append( results['gt_labels'][i] )

                # clw note: 改为泊松融合
                # if (x2 - x1) % 2 == 0:
                #     x2+=1
                # if (y2-y1)%2==0:
                #     y2+=1
                obj = src_img[y1:y2, x1:x2, :]

                #mask = 255 * np.ones(obj.shape, obj.dtype)
                mask = 255 * np.ones(obj.shape, obj.dtype)
                #mask = np.zeros(obj.shape[:2], obj.dtype)
                h, w = mask.shape[:2]
                # rect of thickness > 1
                #cv2.rectangle(mask, (0, 0), (w, h), 255, 1)  # the clone will now aligns from the centre, thick=1 is not work

                # for i in range(obj.shape[0]):
                #     mask[0][i] = 255

                # mask = np.zeros_like(obj)  # clw note: mask = np.zeros(obj.shape[:2]) can cause error   cv2.error: OpenCV(4.5.1) /tmp/pip-req-build-7m_g9lbm/opencv/modules/core/src/matrix_wrap.cpp:1613: error: (-215:Assertion failed) !fixedSize() in function 'release'
                # bound = -3
                # mask[:, bound:] = 255

                center = ( (x2+x1)//2, (y2+y1)//2)
                #print(x1, x2, y1, y2, center, obj.shape, mask.shape, bg_img.shape)
                bg_img = cv2.seamlessClone(obj, bg_img, mask, center, cv2.MIXED_CLONE)  # cv2.NORMAL_CLONE   cv2.MIXED_CLONE

                # bg_img[y1:y2, x1:x2 , :] = src_img[y1:y2, x1:x2, :]
                # mask = cv2.rectangle(mask,(x1,y1), (x2, y2), 255, 6)

            # image = cv2.inpaint(bg_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
            # results['img'] = image

            if len(tmp_gt_bboxes) == 0:
                return results   # 不做融合了,直接返回
            results['gt_bboxes'] = np.array(tmp_gt_bboxes)
            results['gt_labels'] = np.array(tmp_gt_labels)
            #print(tmp_gt_bboxes, tmp_gt_labels)
            results['img'] = bg_img

            ### 可视化确认结果无误
            # filename = results['img_info']['file_name']
            # img_out = results['img'].copy()
            # for box in results['gt_bboxes']:
            #     cv2.rectangle(img_out, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=0, lineType=cv2.LINE_AA)
            #
            # cv2.imwrite('/home/user/' + filename, src_img)
            # cv2.imwrite('/home/user/' + bg_img_fn, bg_img)
            # cv2.imwrite('/home/user/' + filename.split('.')[0] + '_' + bg_img_fn.split('.')[0] + '.jpg', img_out)
            ###

        return results


# import scipy
# from scipy.sparse.linalg import spsolve
# def laplacian_matrix(n, m):
#     """Generate the Poisson matrix.
#
#     Refer to:
#     https://en.wikipedia.org/wiki/Discrete_Poisson_equation
#
#     Note: it's the transpose of the wiki's matrix
#     """
#     mat_D = scipy.sparse.lil_matrix((m, m))
#     mat_D.setdiag(-1, -1)
#     mat_D.setdiag(4)
#     mat_D.setdiag(-1, 1)
#
#     mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
#
#     mat_A.setdiag(-1, 1 * m)
#     mat_A.setdiag(-1, -1 * m)
#
#     return mat_A
#
# def poisson_edit(source, target, mask, offset):
#     """The poisson blending function.
#
#     Refer to:
#     Perez et. al., "Poisson Image Editing", 2003.
#     """
#
#     # Assume:
#     # target is not smaller than source.
#     # shape of mask is same as shape of target.
#     y_max, x_max = target.shape[:-1]
#     y_min, x_min = 0, 0
#
#     x_range = x_max - x_min
#     y_range = y_max - y_min
#
#     M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
#     source = cv2.warpAffine(source, M, (x_range, y_range))
#
#     mask = mask[y_min:y_max, x_min:x_max]
#     mask[mask != 0] = 1
#
#     mat_A = laplacian_matrix(y_range, x_range)
#
#     # for \Delta g
#     laplacian = mat_A.tocsc()
#
#     # set the region outside the mask to identity
#     for y in range(1, y_range - 1):
#         for x in range(1, x_range - 1):
#             if mask[y, x] == 0:
#                 k = x + y * x_range
#                 mat_A[k, k] = 1
#                 mat_A[k, k + 1] = 0
#                 mat_A[k, k - 1] = 0
#                 mat_A[k, k + x_range] = 0
#                 mat_A[k, k - x_range] = 0
#
#     mat_A = mat_A.tocsc()
#
#     mask_flat = mask.flatten()
#     for channel in range(source.shape[2]):
#         source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
#         target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()
#
#         # inside the mask:
#         # \Delta f = div v = \Delta g
#         alpha = 1
#         mat_b = laplacian.dot(source_flat) * alpha
#
#         # outside the mask:
#         # f = t
#         mat_b[mask_flat == 0] = target_flat[mask_flat == 0]
#
#         x = spsolve(mat_A, mat_b)
#         # print(x.shape)
#         x = x.reshape((y_range, x_range))
#         # print(x.shape)
#         x[x > 255] = 255
#         x[x < 0] = 0
#         x = x.astype('uint8')
#
#         target[y_min:y_max, x_min:x_max, channel] = x
#
#     return target