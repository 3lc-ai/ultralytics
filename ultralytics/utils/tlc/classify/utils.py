# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, check_cls_dataset
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.tlc.constants import TLC_COLORSTR
from ultralytics.utils.tlc.utils import check_tlc_dataset

def tlc_check_cls_dataset(
        data: str,
        tables: dict[str, tlc.Table | tlc.Url | Path | str] | None,
        image_column_name: str,
        label_column_name: str,
        project_name: str | None = None,
    ) -> dict[str, tlc.Table | dict[float, str] | int]:
    """ Get or create tables for YOLOv8 classification datasets. data is ignored when tables is provided.
    
    :param data: Path to an ImageFolder dataset
    :param tables: Dictionary of tables, if already created
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :param project_name: Name of the project
    :return: Dictionary of tables and class names, with keys for each split and "names"
    """
    return check_tlc_dataset(
        data,
        tables,
        image_column_name,
        label_column_name,
        dataset_checker=check_cls_dataset,
        table_creator=get_or_create_cls_table,
        project_name=project_name,
        check_backwards_compatible_table_name=False
    )

def get_or_create_cls_table(
        key: str,
        data_dict: dict[str, object],
        image_column_name: str,
        label_column_name: str,
        project_name: str,
        dataset_name: str,
        table_name: str,
    ) -> tlc.Table:
    """ Get or create a classification table from a dataset dictionary.
    
    :param data_dict: Dictionary of dataset information
    :param project_name: Name of the project
    :param dataset_name: Name of the dataset
    :param table_name: Name of the table
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :return: A tlc.Table.from_image_folder() table
    """
    return tlc.Table.from_image_folder(
        root=data_dict[key],
        image_column_name=image_column_name,
        label_column_name=label_column_name,
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        extensions=IMG_FORMATS,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLOv8 integration"
    )