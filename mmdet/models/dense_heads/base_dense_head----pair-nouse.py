from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)  # clw note: self: RPNHead ;  outs: rpn_cls_score and rpn_bbox_pred
                  # ([(8, 3, 384, 384), (8, 3, 192, 192), .... (8, 3, 24, 24)], and [(8, 12, 384, 384), (8, 12, 192, 192) ..., (8, 12, 24, 24)]  )
        rpn_outs_half = []  # ([(4, 3, 384, 384), (4, 3, 192, 192), .... (4, 3, 24, 24)], and [(4, 12, 384, 384), (4, 12, 192, 192) ..., (4, 12, 24, 2
        for outs_0 in outs:
            tmp = []
            for outs_1 in outs_0:
                tmp.append(outs_1[::2, :, :, :])  # clw note: only get origin img rpn out(0,2,4,6) to compute loss, drop template
            rpn_outs_half.append(tmp)

        if gt_labels is None:  # clw note: only 2 class in rpn, so no gt_labels, only gt_bboxes
            #loss_inputs = outs + (gt_bboxes, img_metas)
            loss_inputs = tuple(rpn_outs_half) + (gt_bboxes, img_metas)  # clw modify
        else:
            loss_inputs = tuple(rpn_outs_half) + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            #copy train img_meta to pair. 1,2,3->1,1,2,2,3,3
            img_meta = [img_metas[i//2] for i in range(2*len(img_metas))]  # clw note: ?  TODO
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
