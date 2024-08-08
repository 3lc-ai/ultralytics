from __future__ import annotations

import tlc

from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, check_cls_dataset
from ultralytics.utils import colorstr, LOGGER

def tlc_check_cls_dataset(
        data: str,
        tables: dict[str, tlc.Table | tlc.Url | Path | str] | None,
        image_column_name: str,
        label_column_name: str,
    ) -> dict[str, tlc.Table | dict[float, str] | int]:
    """ Get or create tables for YOLOv8 classification datasets. If tables are provided, data is ignored.
    
    :param data: Path to an ImageFolder dataset
    :param tables: Dictionary of tables, if already created
    :return: Dictionary of tables and class names, with keys for each split and "names"
    """
    if tables is None:
        tables = {}
        # If no tables exist, get the data
        data_dict = check_cls_dataset(data)

        # Get or create tables
        LOGGER.info(f"{colorstr('3LC')}: Creating or reusing tables from {data}")

        for key in ("train", "val", "test"):
            if data_dict.get(key) is not None:
                name = Path(data).name

                table = tlc.Table.from_image_folder(
                    root=data_dict[key],
                    image_column_name=image_column_name,
                    label_column_name=label_column_name,
                    extensions=IMG_FORMATS,
                    table_name="original",
                    dataset_name=f"{name}-{key}",
                    project_name=f"{name}-YOLOv8", # TODO: Allow user to provide this
                    if_exists="reuse",
                    description=f"Original {key} dataset for {data}, created with YOLOv8",
                )

                table.write_to_url()

                # Get the latest version when inferring
                tables[key] = table.latest()

                if tables[key] != table:
                    LOGGER.info(f"   - {key}: Using latest version of table {table.url} -> {tables[key].url}")
                else:
                    LOGGER.info(f"   - {key}: Using original table {tables[key].url}")

    else:
        # Get existing tables if Urls are provided
        LOGGER.info(f"{colorstr('3LC')}: Using data provided through `tables`")
        for key, table in tables.items():
            if isinstance(table, (str, Path, tlc.Url)):
                table_url = tlc.Url(table)
                tables[key] = tlc.Table.from_url(table_url)
            elif isinstance(table, tlc.Table):
                tables[key] = table
            else:
                raise ValueError(
                    f"Invalid type {type(table)} for split {key} provided through `tables`."
                    "Must be a location (string, pathlib.Path or tlc.Url) of a tlc.Table or a tlc.Table object."
                )
            
            LOGGER.info(f"   - {key}: {tables[key].url}")
        
    first_split = next(iter(tables.keys()))
    value_map = tables[first_split].get_value_map(label_column_name)
    names = {k: v["internal_name"] for k, v in value_map.items()}

    return {**tables, "nc": len(names), "names": names}