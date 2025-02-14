# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc
import yaml

from pathlib import Path

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.tlc.constants import TLC_COLORSTR, TLC_PREFIX

from typing import Callable, Iterable


def check_tlc_dataset(
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | str] | None,
    image_column_name: str,
    label_column_name: str,
    dataset_checker: Callable[[str], dict[str, object]] | None = None,
    table_creator: Callable[[str, dict[str, object], str, str, str, str, str], tlc.Table] | None = None,
    table_checker: Callable[[str, tlc.Table], bool] | None = None,
    project_name: str | None = None,
    check_backwards_compatible_table_name: bool = False,
    splits: Iterable[str] | None = None,
) -> dict[str, tlc.Table | dict[float, str] | int]:
    """Get or create tables for YOLOv8 datasets. data is ignored when tables is provided.

    :param data: Path to a dataset
    :param tables: Dictionary of tables, if already created
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :param dataset_checker: Function to check the dataset (yolo implementation, download and checks)
    :param table_creator: Function to create the tables for the YOLO dataset
    :param table_checker: Function to check that a table is compatible with the current task
    :param project_name: Name of the project
    :param check_backwards_compatible_table_name: Whether to check for a backwards compatible table name
    :param splits: List of splits to parse.
    :return: Dictionary of tables and class names
    """
    # If the data starts with the 3LC prefix, parse the YAML file and populate `tables`
    has_prefix = False
    if tables is None and data.startswith(TLC_PREFIX):
        has_prefix = True
        LOGGER.info(f"{TLC_COLORSTR}Parsing 3LC YAML file data={data} and populating tables")
        tables = parse_3lc_yaml_file(data)

    if tables is None:
        tables = {}

        data_dict = dataset_checker(data)

        # Get or create tables
        splits = splits or ("train", "val", "test", "minival")

        for key in splits:
            if data_dict.get(key):
                name = Path(data).stem
                dataset_name = f"{name}-{key}"
                table_name = "initial"

                if project_name is None:
                    project_name = f"{name}-YOLOv8"

                # Previously the table name was "original" and now it is "initial", so we need to check for backwards compatibility
                if check_backwards_compatible_table_name:
                    table_url_backcompatible = tlc.Table._resolve_table_url(
                        table_url=None,
                        root_url=None,
                        project_name=project_name,
                        dataset_name=dataset_name,
                        table_name="original",
                    )

                    if table_url_backcompatible.exists():
                        table_name = "original"

                try:
                    table = table_creator(
                        key,
                        data_dict,
                        image_column_name=image_column_name,
                        label_column_name=label_column_name,
                        project_name=project_name,
                        dataset_name=dataset_name,
                        table_name=table_name,
                    )

                    # Get the latest version when inferring
                    tables[key] = table.latest()

                    if tables[key] != table:
                        LOGGER.info(
                            f"{colorstr(key)}: Using latest version of table from {data}: {table.url} -> {tables[key].url}"
                        )
                    else:
                        LOGGER.info(f"{colorstr(key)}: Using initial version of table from {data}: {tables[key].url}")

                except Exception as e:
                    LOGGER.warning(
                        f"{colorstr(key)}: Failed to read or create table for split {key} from {data}: {e!s}"
                    )

    else:
        # LOGGER.info(f"{TLC_COLORSTR}Using data directly from tables")
        for key, table in tables.items():
            if splits is not None and key not in splits:
                continue

            if isinstance(table, (str, Path, tlc.Url)):
                try:
                    table_url = tlc.Url(table)
                    tables[key] = tlc.Table.from_url(table_url)
                except Exception as e:
                    raise ValueError(
                        f"Error loading table from {table} for split '{key}' provided through `tables`."
                    ) from e
            elif isinstance(table, tlc.Table):
                tables[key] = table
            else:
                msg = (
                    f"Invalid type {type(table)} for split {key} provided through `tables`."
                    "Must be a tlc.Table object or a location (string, pathlib.Path or tlc.Url) of a tlc.Table."
                )

                raise ValueError(msg)

            # Check that the table is compatible with the current task
            if table_checker is not None:
                table_checker(tables[key], image_column_name, label_column_name)

            source = "3LC YAML file" if has_prefix else "provided tables"
            LOGGER.info(f"{colorstr(key)}: Using table {tables[key].url} from {source}")

    first_split = next(iter(tables.keys()))

    value_map = get_table_value_map(tables[first_split], label_column_name)

    names = {int(k): v["internal_name"] for k, v in value_map.items()}

    for split in tables:
        other_value_map = get_table_value_map(tables[split], label_column_name)

        if other_value_map != value_map:
            msg = f"All splits must have the same categories, but {split} has different categories from {first_split}."
            raise ValueError(msg)

    # Map name indices to 0, 1, ..., n-1
    names_yolo = dict(enumerate(names.values()))
    range_to_3lc_class = dict(enumerate(names))

    return {
        **tables,
        "names": names_yolo,
        "names_3lc": value_map,
        "nc": len(names),
        "range_to_3lc_class": range_to_3lc_class,
        "3lc_class_to_range": {v: k for k, v in range_to_3lc_class.items()},
    }


def get_table_value_map(table: tlc.Table, label_column_name: str) -> dict[int, dict[str, object]]:
    """Get the value map for a table.

    :param table: The table to get the value map for.
    :param label_column_name: The name of the label column.
    :returns: The value map for the table.
    """
    value_map = table.get_value_map(label_column_name)
    if value_map is None:
        value_map = (
            table.schema.values["rows"].values["segmentations"].values["instance_properties"].values["label"].value.map
        )

    return value_map


def parse_3lc_yaml_file(data_file: str) -> dict[str, tlc.Table]:
    """Parse a 3LC YAML file and return the corresponding tables.

    :param data_file: The path to the 3LC YAML file.
    :returns: The tables pointed to by the YAML file.
    """
    # Read the YAML file, removing the prefix
    if not (data_file_url := tlc.Url(data_file.replace(TLC_PREFIX, ""))).exists():
        raise FileNotFoundError(f"Could not find YAML file {data_file_url}")

    data_config = yaml.safe_load(data_file_url.read())

    path = data_config.get("path")
    splits = [key for key in data_config if key != "path"]

    tables = {}
    for split in splits:
        # Handle :latest at the end
        if data_config[split].endswith(":latest"):
            latest = True
            split_path = data_config[split][: -len(":latest")]
        else:
            latest = False
            split_path = data_config[split]

        if split_path.startswith("./"):
            LOGGER.debug(f"{TLC_COLORSTR}{split} split path starts with './', removing it.")
            split_path = split_path[2:]

        table_url = tlc.Url(path) / split_path if path else tlc.Url(split_path)

        table = tlc.Table.from_url(table_url)

        if latest:
            table = table.latest()

        tables[split] = table

    return tables
