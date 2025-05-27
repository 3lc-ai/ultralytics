# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import numpy as np
import tlc
from typing import Any, Callable, Literal

from tlc.core.builtins.types.bounding_box import CenteredXYWHBoundingBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import segments2boxes
from ultralytics.utils.tlc.engine.dataset import TLCDatasetMixin
from ultralytics.utils import LOGGER

TableFormat = Literal["detect", "segment"]
SegmentType = Literal["absolute", "relative"]


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
        image_column_name=None,
        label_column_name=None,
        **kwargs,
    ):
        """3LC equivalent of YOLODataset, populating the data fields from a 3LC Table.

        :param table: The 3LC table containing the dataset
        :param data: Optional data parameter for YOLODataset
        :param task: Either "segment" or "detect"
        :param exclude_zero: Whether to exclude zero-class annotations
        :param class_map: Optional mapping from original class indices to new ones
        :param image_column_name: Name of the image column in the table
        :param label_column_name: Name of the label column in the table
        """
        assert task in ("segment", "detect"), (
            f"Unsupported task: {task} for TLCYOLODataset. Only 'segment' and 'detect' are supported."
        )
        self.table = table
        self._exclude_zero = exclude_zero
        self._class_map = class_map if class_map is not None else IdentityDict()
        self._image_column_name = image_column_name
        self._label_column_name = label_column_name
        self._task = task

        # Initialize format-related properties
        self._table_format: TableFormat = "detect"  # Will be set by _infer_table_format
        self._detection_factory: Callable[[list[float]], tlc.BoundingBox] | None = None
        self._segment_type: SegmentType | None = None

        self._infer_table_format(table)
        self._verify_table_format_task_compatibility()

        super().__init__(table, data=data, task=task, **kwargs)
        self._post_init()

    def _infer_table_format(self, table: tlc.Table) -> None:
        """Infer the format of the table and set appropriate format properties.

        This method determines whether the table contains detection or segmentation data
        and sets the corresponding format properties.
        """
        column_name, instances_name, label_name = self._label_column_name.split(".")

        # Detection
        try:
            self._detection_factory = tlc.BoundingBox.from_schema(
                table.rows_schema.values[column_name].values[instances_name]
            )
            self._table_format = "detect"
            return
        except Exception as e:
            LOGGER.debug(f"Table {table.url} is not a detection table: {e}")

        # Segmentation
        try:
            rles_schema_value = table.rows_schema.values[column_name].values[
                instances_name
            ]
            self._segment_type = (
                "relative"
                if getattr(rles_schema_value, "polygons_are_relative", False)
                else "absolute"
            )
            self._table_format = "segment"
            return
        except Exception as e:
            LOGGER.debug(f"Table {table.url} is not a segmentation table: {e}")

        raise ValueError(f"Table {table.url} is not a detection or segmentation table.")

    def _verify_table_format_task_compatibility(self) -> None:
        """Verify that the table format is compatible with the task.

        Raises:
            ValueError: If the table format is not compatible with the task.
        """
        if self._task == "segment" and self._table_format != "segment":
            raise ValueError(
                f"Table {self.table.url} is not a segmentation table, but the task is set to 'segment'."
            )

        if self._task == "detect" and self._table_format == "segment":
            LOGGER.debug(
                f"Table {self.table.url} is a segmentation table, using for detection."
            )

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

    def _get_label_from_row(
        self, im_file: str, row: Any, example_id: int
    ) -> dict[str, Any]:
        """Get the label for a row in the appropriate format.

        Args:
            im_file: Path to the image file
            row: The table row to process
            example_id: The index of the row in the table

        Returns:
            A dictionary containing the label information in YOLO format
        """
        if self._table_format == "detect":
            return tlc_table_row_to_yolo_label(
                row,
                self._detection_factory,
                self._class_map,
                im_file,
                label_column_name=self._label_column_name,
            )
        else:  # segment
            return tlc_table_row_to_segment_label(
                self.table[example_id],
                self._segment_type,
                self._class_map,
                im_file,
                self._label_column_name,
                row_index=example_id,
            )

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


def convert_to_xywh(
    bbox: tlc.BoundingBox, image_width: int, image_height: int
) -> CenteredXYWHBoundingBox:
    if isinstance(bbox, CenteredXYWHBoundingBox):
        return bbox
    else:
        return CenteredXYWHBoundingBox.from_top_left_xywh(
            bbox.to_top_left_xywh().normalize(image_width, image_height)
        )


def unpack_box(
    bbox: dict[str, int | float],
    table_format: Callable[[list[float]], tlc.BoundingBox],
    image_width: int,
    image_height: int,
    label_key: str,
) -> tuple[int, list[float]]:
    coordinates = [bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]
    return bbox[label_key], convert_to_xywh(
        table_format(coordinates), image_width, image_height
    )


def unpack_boxes(
    bboxes: list[dict[str, int | float]],
    class_map: dict[int, int],
    table_format: Callable[[list[float]], tlc.BoundingBox],
    image_width: int,
    image_height: int,
    label_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    classes_list, boxes_list = [], []
    for bbox in bboxes:
        _class, box = unpack_box(
            bbox, table_format, image_width, image_height, label_key
        )

        # Ignore boxes with non-positive width or height
        if box[2] > 0 and box[3] > 0:
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
    row,
    detection_factory: Callable[[list[float]], tlc.BoundingBox],
    class_map: dict[int, int],
    im_file: str,
    label_column_name: str,
) -> dict[str, Any]:
    """Convert a table row from a 3lc Table to a Ultralytics YOLO label dict.

    Args:
        row: The table row to convert
        detection_factory: Factory function to create bounding boxes
        class_map: A dictionary mapping 3lc class labels to contiguous class labels
        im_file: The path to the image file of the row
        label_column_name: The name of the label column in the table

    Returns:
        A dictionary containing the Ultralytics YOLO label information
    """
    bounding_boxes_column_key, bounding_boxes_list_key, label_key = (
        label_column_name.split(".")
    )

    classes, bboxes = unpack_boxes(
        row[bounding_boxes_column_key][bounding_boxes_list_key],
        class_map,
        detection_factory,
        row[bounding_boxes_column_key][tlc.IMAGE_WIDTH],
        row[bounding_boxes_column_key][tlc.IMAGE_HEIGHT],
        label_key,
    )

    return dict(
        im_file=im_file,
        shape=(
            row[bounding_boxes_column_key][tlc.IMAGE_HEIGHT],
            row[bounding_boxes_column_key][tlc.IMAGE_WIDTH],
        ),  # format: (height, width)
        cls=classes,
        bboxes=bboxes,
        segments=[],
        keypoints=None,
        normalized=True,
        bbox_format="xywh",
    )


def tlc_table_row_to_segment_label(
    row,
    segment_type: SegmentType,
    class_map: dict[int, int],
    im_file: str,
    label_column_name: str,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Convert a table row from a 3lc Table to a Ultralytics YOLO segmentation label dict.

    Args:
        row: The table row to convert
        segment_type: Whether the segments are absolute or relative coordinates
        class_map: A dictionary mapping 3lc class labels to contiguous class labels
        im_file: The path to the image file of the row
        label_column_name: The name of the label column in the table
        row_index: Optional index of the row for logging purposes

    Returns:
        A dictionary containing the Ultralytics YOLO segmentation label information
    """
    segmentations_column_key, instance_properties_key, label_key = (
        label_column_name.split(".")
    )

    segmentations = row[segmentations_column_key]
    # Get image size
    height, width = segmentations[tlc.IMAGE_HEIGHT], segmentations[tlc.IMAGE_WIDTH]

    classes = []
    segments = []

    for i, (category, polygon) in enumerate(
        zip(
            segmentations[instance_properties_key][label_key],
            segmentations["polygons"],
        )
    ):
        # Handle polygons with zero area
        if len(polygon) < 6:
            LOGGER.warning(
                f"Polygon {i} in row {row_index} has fewer than 6 points and will be ignored."
            )
            continue

        classes.append(class_map[category])
        row_segments = np.array(polygon).reshape(-1, 2)

        if segment_type == "absolute":
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
