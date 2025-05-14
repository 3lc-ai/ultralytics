# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from ultralytics.data.dataset import ClassificationDataset
from ultralytics.utils.tlc.engine.dataset import TLCDatasetMixin

from typing import Any


class TLCClassificationDataset(TLCDatasetMixin, ClassificationDataset):
    """
    Initialize 3LC classification dataset for use in YOLO classification.

    Args:
        table (tlc.Table): The 3LC table with classification data. Needs columns 'image' and 'label'.
        args (Namespace): See parent.
        augment (bool): See parent.
        prefix (str): See parent.

    """

    def __init__(
        self,
        table,
        args,
        augment=False,
        prefix="",
        image_column_name=tlc.IMAGE,
        label_column_name=tlc.LABEL,
        exclude_zero=False,
        class_map=None,
    ):
        # Populate self.samples with image paths and labels
        # Each is a tuple of (image_path, label)
        assert isinstance(table, tlc.Table)
        self.table = table
        self.root = table.url
        self.prefix = prefix
        self._image_column_name = image_column_name
        self._label_column_name = label_column_name
        self._exclude_zero = exclude_zero
        self._class_map = class_map

        self.verify_schema()

        example_ids, im_files, labels = self._get_rows_from_table()

        self.example_ids = example_ids
        self.samples = list(zip(im_files, labels))

        # Initialize attributes (e.g. transforms)
        self._init_attributes(args, augment, prefix)

        # Call mixin
        self._post_init()

    def verify_schema(self):
        """Verify that the provided Table has the desired entries"""

        # Check for data in columns
        assert len(self.table) > 0, f"Table {self.root.to_str()} has no rows."
        first_row = self.table.table_rows[0]
        assert isinstance(first_row[self._image_column_name], str), (
            f"First value in image column '{self._image_column_name}' in table {self.root.to_str()} is not a string."
        )
        assert isinstance(first_row[self._label_column_name], int), (
            f"First value in label column '{self._label_column_name}' in table {self.root.to_str()} is not an integer."
        )

    def verify_images(self):
        """Called by parent init_attributes, but this is handled by the 3LC mixin."""
        return self.samples

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> Any:
        label = row[self._label_column_name]

        if self._class_map:
            label = self._class_map[label]

        return label
