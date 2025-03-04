# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

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

