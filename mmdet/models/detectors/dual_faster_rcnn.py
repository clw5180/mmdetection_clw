from ..builder import DETECTORS
from .two_stage import TwoStageDetector

import torch
import torch.nn as nn

from mmdet.models.utils import Scale, Scale_channel


@DETECTORS.register_module()
class DualFasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 dual_train=True,
                 dual_test=True,
                 # template_train=False,
                 style='sub_feat'):
        super(DualFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.dual_train = dual_train
        self.dual_test = dual_test
        # self.template_train = template_train
        self.style = style
        if self.style == 'sub_feat':
            self.scale_a = Scale(0.5)
            self.scale_b = Scale(0.5)
        elif self.style == 'vector_add_feat':
            self.scale_a = Scale_channel(1, 256)
            self.scale_b = Scale_channel(1, 256)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat_dual(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def extract_feat_dual(self, img):
        if self.dual_train is True and self.style is not None:

            b, c, h, w = img.shape
            img = img.reshape(-1, c // 2, h, w) #一张（1，6，x, x ）输入分成瑕疵图和模板图
            if self.style == 'sub_img':
                img = img[0::2, :, :, :] - img[1::2, :, :, :]
                x = self.extract_feat(img)
            elif self.style == 'add_img':
                img = img[0::2, :, :, :] + img[1::2, :, :, :]
                x = self.extract_feat(img)
            elif self.style == 'sub_feat':
                x = self.extract_feat(img)
                x_ = []

                for i, lvl_feat in enumerate(x):
                    x_.append(self.scale_a(lvl_feat[0::2, :, :, :]) + self.scale_b(lvl_feat[1::2, :, :, :])) #瑕疵图和模板图融合
                x = tuple(x_)
            elif self.style == 'add_feat':
                x = self.extract_feat(img)
                x_ = []
                for lvl_feat in x:
                    x_.append(lvl_feat[0::2, :, :, :] + lvl_feat[1::2, :, :, :])
                x = tuple(x_)
            elif self.style == 'vector_add_feat':
                x = self.extract_feat(img)
                x_ = []
                for i, lvl_feat in enumerate(x):
                    x_.append(self.scale_a(lvl_feat[0::2, :, :, :]) + self.scale_b(lvl_feat[1::2, :, :, :]))
                x = tuple(x_)
        else:
            x = self.extract_feat(img)

        return x

    #base.py
    def extract_feats_dual(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat_dual(img) for img in imgs]

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.dual_test:
            x = self.extract_feat_dual(img)
        else:
            x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.dual_test:
            x = self.extract_feat_dual(img)
        else:
            x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.dual_test:
            x = self.extract_feats_dual(imgs)
        else:
            x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)


