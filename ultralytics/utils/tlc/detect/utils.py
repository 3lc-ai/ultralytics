# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from pathlib import Path

import tlc
import yaml
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (
    _TLCPredictedBoundingBox,
    _TLCPredictedBoundingBoxes,
)

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.tlc.constants import TLC_COLORSTR, TLC_PREFIX
from ultralytics.utils.tlc.detect.dataset import TLCYOLODataset
from ultralytics.utils.tlc.utils import check_tlc_dataset

def tlc_check_det_dataset(
        data: str,
        tables: dict[str, tlc.Table | tlc.Url | Path | str] | None,
        image_column_name: str,
        label_column_name: str,
        project_name: str | None = None,
    ) -> dict[str, tlc.Table | dict[float, str] | int]:
    # If the data starts with the 3LC prefix, parse the YAML file and populate `tables`
    if tables is None and data.startswith(TLC_PREFIX):
        tables = parse_3lc_yaml_file(data)

    return check_tlc_dataset(
        data,
        tables,
        image_column_name,
        label_column_name,
        dataset_checker=check_det_dataset,
        table_creator=get_or_create_det_table,
        table_checker=check_det_table,
        project_name=project_name,
        check_backwards_compatible_table_name=True,
    )
    
def get_or_create_det_table(
    key: str,
    data_dict: dict[str, object],
    image_column_name: str,
    label_column_name: str,
    project_name: str,
    dataset_name: str,
    table_name: str,
) -> tlc.Table:
    """ Get or create a detection table from a dataset dictionary.

    :param data_dict: Dictionary of dataset information
    :param project_name: Name of the project
    :param dataset_name: Name of the dataset
    :param table_name: Name of the table
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :return: A tlc.Table.from_image_folder() table
    """
    return tlc.Table.from_yolo(
        dataset_yaml_file=data_dict["yaml_file"],
        split=key,
        override_split_path=data_dict[key],
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLOv8 integration"
    )

def build_tlc_yolo_dataset(
        cfg,
        table,
        batch,
        data,
        mode="train",
        rect=False,
        stride=32,
        multi_modal=False,
        settings=None,):
    if multi_modal:
        return ValueError("Multi-modal datasets are not supported in the 3LC YOLOv8 integration.")
    
    if mode=="train":
        sampling_weights = settings.sampling_weights
        exclude_zero_weight = settings.exclude_zero_weight_training
    else:
        sampling_weights = False # Never use sampling weights for validation
        exclude_zero_weight = settings.exclude_zero_weight_collection

    return TLCYOLODataset(
        table,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        sampling_weights=sampling_weights,
        exclude_zero_weight=exclude_zero_weight,
    )

def parse_3lc_yaml_file(data_file: str) -> dict[str, tlc.Table]:
    """ Parse a 3LC YAML file and return the corresponding tables.
    
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
            split_path = data_config[split][:-len(":latest")]
        else:
            latest = False
            split_path = data_config[split]

        if split_path.startswith("./"):
            LOGGER.debug(f"{TLC_COLORSTR}{split} split path starts with './', removing it.")
            split_path = split_path[2:]
        elif split_path.startswith("/"):
            LOGGER.debug(f"{TLC_COLORSTR}{split} split path starts with '/', removing it.")
            split_path = split_path[1:]

        table_url = tlc.Url(path) / split_path if path else tlc.Url(split_path)

        table = tlc.Table.from_url(table_url)

        if latest:
            table = table.latest()

        tables[split] = table

    return tables

def check_det_table(split: str, table: tlc.Table):
    """ Check that a table is compatible with the detection task in the 3LC YOLOv8 integration.
    
    :param split: The split of the table.
    :param table: The table to check.
    :raises: ValueError if the table is not compatible with the detection task.
    """
    return

def yolo_predicted_bounding_box_schema(categories: dict[int, str]) -> tlc.Schema:
    """ Create a 3LC bounding box schema for YOLOv8

    :param categories: Categories for the current dataset.
    :returns: The YOLO bounding box schema for predicted boxes.
    """
    label_value_map = {float(i): tlc.MapElement(class_name) for i, class_name in categories.items()}

    bounding_box_schema = tlc.BoundingBoxListSchema(
        label_value_map=label_value_map,
        x0_number_role=tlc.NUMBER_ROLE_BB_CENTER_X,
        x1_number_role=tlc.NUMBER_ROLE_BB_SIZE_X,
        y0_number_role=tlc.NUMBER_ROLE_BB_CENTER_Y,
        y1_number_role=tlc.NUMBER_ROLE_BB_SIZE_Y,
        x0_unit=tlc.UNIT_RELATIVE,
        y0_unit=tlc.UNIT_RELATIVE,
        x1_unit=tlc.UNIT_RELATIVE,
        y1_unit=tlc.UNIT_RELATIVE,
        description='Predicted Bounding Boxes',
        writable=False,
        is_prediction=True,
        include_segmentation=False,
    )

    return bounding_box_schema


def yolo_loss_schemas() -> dict[str, tlc.Schema]:
    """ Create a 3LC schema for YOLOv8 loss metrics.

    :returns: The YOLO loss schemas.
    """
    schemas = {}
    schemas['loss'] = tlc.Schema(description='Sample loss',
                                 writable=False,
                                 value=tlc.Float32Value(),
                                 display_importance=3003)
    schemas['box_loss'] = tlc.Schema(description='Box loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3004)
    schemas['obj_loss'] = tlc.Schema(description='Object loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3005)
    schemas['cls_loss'] = tlc.Schema(description='Classification loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3006)
    return schemas

def construct_bbox_struct(
    predicted_annotations: list[dict[str, int | float | dict[str, float]]],
    image_width: int,
    image_height: int,
    inverse_label_mapping: dict[int, int] | None = None,
) -> _TLCPredictedBoundingBoxes:
    """Construct a 3LC bounding box struct from a list of bounding boxes.

    :param predicted_annotations: A list of predicted bounding boxes.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param inverse_label_mapping: A mapping from predicted label to category id.
    """

    bbox_struct = _TLCPredictedBoundingBoxes(
        bb_list=[],
        image_width=image_width,
        image_height=image_height,
    )

    for pred in predicted_annotations:
        bbox, label, score, iou = pred['bbox'], pred['category_id'], pred['score'], pred['iou']
        label_val = inverse_label_mapping[label] if inverse_label_mapping is not None else label
        bbox_struct['bb_list'].append(
            _TLCPredictedBoundingBox(
                label=label_val,
                confidence=score,
                iou=iou,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
            ))

    return bbox_struct

def infer_table_format(table: tlc.Table) -> str:
    """ Infer the format of a table.

    :param table: The table to infer the format of.
    :returns: The format of the table.
    """
    if _check_if_yolo_table(table):
        return "YOLO"
    elif _check_if_coco_table(table):
        return "COCO"
    else:
        raise ValueError(f'Table {table.url} is not compatible with YOLOv8, needs to be a YOLO or COCO table.')

def _check_if_yolo_table(table: tlc.Table) -> tuple[bool, str]:
    """Check if the table is a YOLO table.

    :param table: The table to check.
    :returns: True if the table is a YOLO table, False otherwise.
    """
    row_schema = table.row_schema.values

    try:
        assert tlc.IMAGE in row_schema
        assert tlc.WIDTH in row_schema
        assert tlc.HEIGHT in row_schema
        assert tlc.BOUNDING_BOXES in row_schema
        assert tlc.BOUNDING_BOX_LIST in row_schema[tlc.BOUNDING_BOXES].values
        assert tlc.SAMPLE_WEIGHT in row_schema
        assert tlc.LABEL in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values

        X0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X0]
        Y0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y0]
        X1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X1]
        Y1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y1]

        assert X0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_X
        assert Y0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_Y
        assert X1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_X
        assert Y1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_Y

        assert X0.value.unit == tlc.UNIT_RELATIVE
        assert Y0.value.unit == tlc.UNIT_RELATIVE
        assert X1.value.unit == tlc.UNIT_RELATIVE
        assert Y1.value.unit == tlc.UNIT_RELATIVE

    except AssertionError:
        return False

    return True

def _check_if_coco_table(table: tlc.Table) -> bool:
    """Check if the table is a COCO table.

    :param table: The table to check.
    :returns: True if the table is a COCO table, False otherwise.
    """
    row_schema = table.row_schema.values

    try:
        assert tlc.IMAGE in row_schema
        assert tlc.WIDTH in row_schema
        assert tlc.HEIGHT in row_schema
        assert tlc.BOUNDING_BOXES in row_schema
        assert tlc.BOUNDING_BOX_LIST in row_schema[tlc.BOUNDING_BOXES].values
        assert tlc.SAMPLE_WEIGHT in row_schema
        assert tlc.LABEL in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values

        X0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X0]
        Y0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y0]
        X1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X1]
        Y1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y1]

        assert X0.value.number_role == tlc.NUMBER_ROLE_BB_MIN_X
        assert Y0.value.number_role == tlc.NUMBER_ROLE_BB_MIN_Y
        assert X1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_X
        assert Y1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_Y

    except AssertionError:
        return False

    return True


def write_3lc_yaml(data_file: str, tables: dict[str, tlc.Table]):
    """ Write a 3LC YAML file for the given tables.

    :param data_file: The path to the original YOLO YAML file.
    :param tables: The 3LC tables.
    """
    new_yaml_url = tlc.Url(data_file.replace('.yaml', '_3lc.yaml'))
    if new_yaml_url.exists():
        LOGGER.info(f'{TLC_COLORSTR}3LC YAML file already exists: {str(new_yaml_url)}. To use this file,'
                    f' add a 3LC prefix: "3LC://{str(new_yaml_url)}".')
        return

    # Common path for train, val, test tables:
    #                                        v           <--          <--          *
    # projects / yolov8-<dataset_name> / datasets / <dataset_name> / tables / <table_url> / files
    path = tables['train'].url.parent.parent.parent

    # Get relative paths for each table to write to YAML file
    split_paths = {split: str(tlc.Url.relative_from(tables[split].url, path).apply_aliases()) for split in tables}

    # Add :latest to each
    split_paths_latest = {split: f'{path}:latest' for split, path in split_paths.items()}

    # Create 3LC yaml file
    data_config = {'path': str(path), **split_paths_latest}
    new_yaml_url.write(yaml.dump(data_config, sort_keys=False, encoding='utf-8'))

    LOGGER.info(f'{TLC_COLORSTR}Created 3LC YAML file: {str(new_yaml_url)}. To use this file,'
                f' add a 3LC prefix: "3LC://{str(new_yaml_url)}".')
