import tlc

from pathlib import Path

import torch
from ultralytics.utils.metrics import box_iou, mask_iou

import torch.nn.functional as F

from typing import Iterable
from ultralytics.utils.tlc.utils import check_tlc_dataset
from ultralytics.data.utils import check_det_dataset
def tlc_check_seg_dataset(
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | Path | str] | None,
    image_column_name: str,
    label_column_name: str,
    project_name: str | None = None,
    splits: Iterable[str] | None = None,
) -> dict[str, tlc.Table | dict[float, str] | int]:
    return check_tlc_dataset(
        data,
        tables,
        image_column_name,
        label_column_name,
        dataset_checker=check_det_dataset,
        table_creator=get_or_create_seg_table,
        table_checker=check_seg_table,
        project_name=project_name,
        check_backwards_compatible_table_name=True,
        splits=splits,
    )

def get_or_create_seg_table(
        key: str,
        data_dict: dict[str, object],
        image_column_name: str,
        label_column_name: str,
        project_name: str,
        dataset_name: str,
        table_name: str,
    ) -> tlc.Table:
    return tlc.Table.from_yolo(
        dataset_yaml_file=data_dict["yaml_file"],
        split=key,
        override_split_path=data_dict[key],
        task="segment",
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLO integration",
    )

def check_seg_table(table: tlc.Table, _0: str, _1: str):
    return None

def compute_mask_iou(gt_cls, pred_masks, gt_masks, overlap=False):
    """Compute IoU for predicted masks against ground truth masks.
    
    Args:
        gt_cls (torch.Tensor): Ground truth class indices.
        pred_masks (torch.Tensor): Predicted masks.
        gt_masks (torch.Tensor): Ground truth masks.
        overlap (bool): Whether to consider overlapping masks.

    Returns:
        torch.Tensor: IoU values.
    """
    if overlap:
        nl = len(gt_cls)
        index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
        gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
        gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
    if gt_masks.shape[1:] != pred_masks.shape[1:]:
        gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
        gt_masks = gt_masks.gt_(0.5)
    return mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))

def compute_box_iou(detections, gt_bboxes):
    """Compute IoU for predicted bounding boxes against ground truth bounding boxes.
    
    Args:
        detections (torch.Tensor): Predicted bounding boxes.
        gt_bboxes (torch.Tensor): Ground truth bounding boxes.
    """
    return box_iou(gt_bboxes, detections[:, :4])