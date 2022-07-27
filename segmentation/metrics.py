import torch
import numpy as np
from typing import Tuple

# pixel miou
def get_iou2(
    labels: torch.Tensor, gt: torch.Tensor, nclasses: int
) -> Tuple[float, torch.Tensor]:
    conf_matrix = torch.zeros((nclasses, nclasses)).long()

    gt_exist = [False] * nclasses
    for cls in range(nclasses):
        cls_mask = gt == cls
        if (cls_mask > 0).any():
            gt_exist[cls] = True
        vals, counts = torch.unique(labels[cls_mask], return_counts=True)
        conf_matrix[cls, vals] += counts

    tp = torch.diagonal(conf_matrix, 0)
    fn = torch.sum(conf_matrix, dim=0)
    fp = torch.sum(conf_matrix, dim=1)
    iou_per_class = tp / (fn + fp - tp + 1e-10)
    return iou_per_class.mean(), iou_per_class


# pixel class accuracy
def get_cls_acc(labels: torch.Tensor, gt: torch.Tensor, nclasses: int) -> torch.Tensor:
    cls_acc = torch.zeros(nclasses)
    for cls in range(nclasses):
        gt_cls_mask = gt == cls
        pred_cls_mask = labels == cls

        cls_acc[cls] = ((pred_cls_mask == gt_cls_mask) & (gt_cls_mask)).sum() / (
            gt_cls_mask.sum() + 1e-10
        )
    return cls_acc

