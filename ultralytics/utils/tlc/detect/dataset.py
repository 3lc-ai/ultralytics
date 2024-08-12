from __future__ import annotations

import numpy as np
import tlc

from ultralytics.utils import ops

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.tlc.engine.dataset import TLCDatasetMixin

from typing import Any

class TLCYOLODataset(TLCDatasetMixin, YOLODataset):
    def __init__(self, table, data=None, task="detect", exclude_zero_weight=None, sampling_weights=None, **kwargs):
        assert task == "detect", f"Unsupported task: {task} for TLCYOLODataset. Only 'detect' is supported."
        self.table = table
        self.display_name = table.dataset_name

        from ultralytics.utils.tlc.detect.utils import infer_table_format
        self._table_format = infer_table_format(table)
        self._exclude_zero_weight=exclude_zero_weight

        self.example_ids = []

        super().__init__(table, data=data, task=task, **kwargs)

        self._post_init(sampling_weights=sampling_weights)

        # TODO: Inspect caching mechanism! Make sure it isn't called unless we want to.

    def get_img_files(self, _):
        return [
            tlc.Url(sample[tlc.IMAGE]).to_absolute().to_str()
            for _, sample
            in self._get_enumerated_table_rows(exclude_zero_weight=self._exclude_zero_weight)
        ]

    def get_labels(self):
        labels = []

        rows = self._get_enumerated_table_rows(exclude_zero_weight=self._exclude_zero_weight)
        for example_id, row in rows:
            self.example_ids.append(example_id)

            labels.append(tlc_table_row_to_yolo_label(row, self._table_format))

        self.example_ids = np.array(self.example_ids, dtype=np.int32)

        return labels
    
    def set_rectangle(self):
        """Save the batch shapes and inidices for the dataset. """
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

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image
    
def unpack_box(bbox: dict[str, int | float]) -> tuple[int | float]:
    return bbox[tlc.LABEL], [bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]

def unpack_boxes(bboxes: list[dict[str, int | float]]) -> tuple[np.ndarray, np.ndarray]:
    classes_list, boxes_list = [], []
    for bbox in bboxes:
        _class, box = unpack_box(bbox)
        classes_list.append(_class)
        boxes_list.append(box)

    # Convert to np array
    boxes = np.array(boxes_list, ndmin=2, dtype=np.float32)
    if len(boxes_list) == 0:
        boxes = boxes.reshape(0, 4)

    classes = np.array(classes_list, dtype=np.float32).reshape((-1, 1))
    assert classes.shape == (boxes.shape[0], 1)
    return classes, boxes

def tlc_table_row_to_yolo_label(row, table_format: str) -> dict[str, Any]:
    classes, bboxes = unpack_boxes(row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST])
    
    if table_format == "COCO":
        # Convert from ltwh absolute to xywh relative
        bboxes_xyxy = ops.ltwh2xyxy(bboxes)
        bboxes = ops.xyxy2xywhn(bboxes_xyxy, w=row['width'], h=row['height'])

    return dict(
        im_file=tlc.Url(row[tlc.IMAGE]).to_absolute().to_str(),
        shape=(row['height'], row['width']),  # format: (height, width)
        cls=classes,
        bboxes=bboxes,
        segments=[],
        keypoints=None,
        normalized=True,
        bbox_format="xywh",
    )