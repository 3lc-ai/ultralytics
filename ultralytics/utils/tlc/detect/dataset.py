# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import numpy as np
import tlc

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import (
    segments2boxes,
)
from ultralytics.utils import ops
from ultralytics.utils.tlc.engine.dataset import TLCDatasetMixin

from ultralytics.utils import LOGGER

from typing import Any


class IdentityDict(dict):
    def __missing__(self, key):
        return key


class TLCYOLODataset(TLCDatasetMixin, YOLODataset):
    def __init__(
        self,
        table,
        data=None,
        task="detect",
        exclude_zero=False,
        class_map=None,
        **kwargs,
    ):
        """3LC equivalent of YOLODataset, populating the data fields from a 3LC Table."""
        assert task in ("segment", "detect"), (
            f"Unsupported task: {task} for TLCYOLODataset. Only 'segment' and 'detect' are supported."
        )
        self.table = table
        self._exclude_zero = exclude_zero
        self._class_map = class_map if class_map is not None else IdentityDict()

        if task == "detect":
            from ultralytics.utils.tlc.detect.utils import is_coco_table, is_yolo_table

            if is_yolo_table(self.table):
                self._table_format = "YOLO"
            elif is_coco_table(self.table):
                self._table_format = "COCO"
            else:
                raise ValueError(f"Unsupported table format for table {table.url}")
        else:
            # The default is absolute, so if it is not present in the schema, it is absolute
            rles_schema_value = (
                self.table.rows_schema.values["segmentations"].values["rles"].value
            )
            polygons_are_relative = getattr(
                rles_schema_value, "polygons_are_relative", False
            )

            self._table_format = (
                "segment_relative" if polygons_are_relative else "segment_absolute"
            )

        super().__init__(table, data=data, task=task, **kwargs)

        self._post_init()

    def get_img_files(self, _):
        """Images are read in `get_labels` to avoid two loops, return empty list here."""
        return []

    def get_labels(self):
        """Get the labels from the table."""
        example_ids, im_files, labels = self._get_rows_from_table()
        self.labels = labels
        self.im_files = im_files
        self.example_ids = np.array(example_ids, dtype=np.int32)

        return self.labels

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> Any:
        if self._table_format in ("COCO", "YOLO"):
            return tlc_table_row_to_yolo_label(
                row, self._table_format, self._class_map, im_file
            )
        elif self._table_format in ("segment_relative", "segment_absolute"):
            return tlc_table_row_to_segment_label(
                self.table[example_id],
                self._table_format,
                self._class_map,
                im_file,
                row_index=example_id,
            )
        else:
            raise ValueError(f"Unsupported table format: {self._table_format}")

    def set_rectangle(self):
        """Save the batch shapes and indices for the dataset."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.example_ids = self.example_ids[irect]

        ar = ar[irect]
        self.irect = irect.copy()

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int)
            * self.stride
        )
        self.batch = bi  # batch index of image


def unpack_box(bbox: dict[str, int | float]) -> tuple[int | float]:
    return bbox[tlc.LABEL], [bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]


def unpack_boxes(
    bboxes: list[dict[str, int | float]], class_map: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    classes_list, boxes_list = [], []
    for bbox in bboxes:
        _class, box = unpack_box(bbox)

        # Ignore boxes with non-positive width or height
        if bbox[2] > 0 and bbox[3] > 0:
            classes_list.append(class_map[_class])
            boxes_list.append(box)

    # Convert to np array
    boxes = np.array(boxes_list, ndmin=2, dtype=np.float32)
    if len(boxes_list) == 0:
        boxes = boxes.reshape(0, 4)

    classes = np.array(classes_list, dtype=np.float32).reshape((-1, 1))
    assert classes.shape == (boxes.shape[0], 1)
    return classes, boxes


def tlc_table_row_to_yolo_label(
    row, table_format: str, class_map: dict[int, int], im_file: str
) -> dict[str, Any]:
    classes, bboxes = unpack_boxes(
        row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST], class_map
    )

    if table_format == "COCO":
        # Convert from ltwh absolute to xywh relative
        bboxes_xyxy = ops.ltwh2xyxy(bboxes)
        bboxes = ops.xyxy2xywhn(bboxes_xyxy, w=row["width"], h=row["height"])

    return dict(
        im_file=im_file,
        shape=(row["height"], row["width"]),  # format: (height, width)
        cls=classes,
        bboxes=bboxes,
        segments=[],
        keypoints=None,
        normalized=True,
        bbox_format="xywh",
    )


def tlc_table_row_to_segment_label(
    row,
    table_format: str,
    class_map: dict[int, int],
    im_file: str,
    row_index: int | None = None,
) -> dict[str, Any]:
    # Row is here in sample view

    segmentations = row["segmentations"]
    # Get image size
    height, width = segmentations["image_height"], segmentations["image_width"]

    classes = []
    segments = []

    for i, (category, polygon) in enumerate(
        zip(segmentations["instance_properties"][tlc.LABEL], segmentations["polygons"])
    ):
        # Handle polygons with zero area
        if len(polygon) < 6:
            LOGGER.warning(
                f"Polygon {i} in row {row_index} has fewer than 6 points and will be ignored."
            )
            continue

        classes.append(class_map[category])
        row_segments = np.array(polygon).reshape(-1, 2)

        if table_format.endswith("absolute"):
            row_segments = row_segments / np.array([width, height])

        segments.append(row_segments)

    # Compute bounding boxes from segments
    if segments:
        bboxes = segments2boxes(segments)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)

    return dict(
        im_file=im_file,
        shape=(height, width),  # format: (height, width)
        cls=np.array(classes).reshape(-1, 1),
        bboxes=bboxes,
        segments=segments,
        keypoints=None,
        normalized=True,
        bbox_format="xywh",
    )
