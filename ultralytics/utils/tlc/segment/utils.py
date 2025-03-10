# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from pathlib import Path

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
        check_backwards_compatible_table_name=False,
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


def check_seg_table(
    table: tlc.Table, image_column_name: str, label_column_name: str
) -> None:
    """Verify that the table is compatible with instance segmentation.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The name of the label column.
    :raises ValueError: If the table is not compatible with instance segmentation.
    """
    row_schema = table.row_schema.values

    # Check that the schema is compatible with instance segmentation
    try:
        # Schema checks
        assert image_column_name in row_schema, (
            f"Image column {image_column_name} not found in table with URL {table.url}"
        )
        assert label_column_name in row_schema, (
            f"Label column {label_column_name} not found in table with URL {table.url}"
        )

        assert hasattr(row_schema[label_column_name], "sample_type"), (
            f"Label column {label_column_name} does not have a sample type"
        )
        sample_type = tlc.SampleType.from_schema(row_schema[label_column_name])
        assert isinstance(sample_type, tlc.InstanceSegmentationPolygons), (
            f"Label column {label_column_name} does not have sample type InstanceSegmentationPolygons"
        )
        assert sample_type.relative, (
            f"Label column {label_column_name} does not have relative coordinates"
        )

    except AssertionError as e:
        msg = f"Schema validation failed for {label_column_name} column in table with URL {table.url}. {str(e)}"
        raise ValueError(msg) from e

    # Check that the table data is compatible with its schema
    try:
        first_row = table[0]
        sample_type.ensure_sample_valid(first_row[label_column_name])
        assert image_column_name in first_row, (
            f"Image column {image_column_name} not found in table with URL {table.url}"
        )

    except (AssertionError, ValueError) as e:
        msg = f"Data validation failed for {label_column_name} column in table with URL {table.url}. {str(e)}"
        raise ValueError(msg) from e
