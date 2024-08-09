import tlc
import json

from multiprocessing.pool import ThreadPool
import numpy as np
from pathlib import Path
from itertools import repeat

from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.utils import verify_image
from ultralytics.utils import LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.tlc.engine.dataset import TLCDatasetMixin


class TLCClassificationDataset(TLCDatasetMixin, ClassificationDataset):
    """
    Initialize 3LC classification dataset for use in YOLO classification.

    Args:
        table (tlc.Table): The 3LC table with classification data. Needs columns 'image' and 'label'.
        args (Namespace): See parent.
        augment (bool): See parent.
        prefix (str): See parent.
    
    """
    def __init__(self, table, args, augment=False, prefix="", image_column_name="Image", label_column_name="Label", exclude_zero_weight=False, **kwargs):
        # Populate self.samples with image paths and labels
        # Each is a tuple of (image_path, label)
        assert isinstance(table, tlc.Table)
        self.table = table
        self.root = table.url
        self.display_name = table.dataset_name

        self.verify_schema(image_column_name, label_column_name)

        self.samples = []
        self.example_ids = []

        iterator = self._get_enumerated_table_rows(exclude_zero_weight=exclude_zero_weight)
        for example_id, row in iterator:
            self.example_ids.append(example_id)
            image_path = Path(tlc.Url(row[image_column_name]).to_absolute().to_str())
            self.samples.append((image_path, row[label_column_name]))

        # Initialize attributes
        self._init_attributes(args, augment, prefix)

        # Call mixin
        super().__init__(**kwargs)

        self._indices = np.arange(len(self.example_ids))
        assert len(self._indices) == len(self.samples)

    def verify_schema(self,image_column_name, label_column_name):
        """ Verify that the provided Table has the desired schema and entries """
        row_schema = self.table.row_schema.values

        # Check for image and label columns in schema
        assert image_column_name in row_schema, f"Image column '{image_column_name}' not found in schema for Table {self.table.url}."
        assert label_column_name in row_schema, f"Label column '{label_column_name}' not found in schema for Table {self.table.url}."

        # Check for desired roles
        assert row_schema[image_column_name].value.string_role == tlc.STRING_ROLE_IMAGE_URL, f"Image column '{image_column_name}' must have role tlc.STRING_ROLE_IMAGE_URL={tlc.STRING_ROLE_IMAGE_URL}."
        assert row_schema[label_column_name].value.number_role == tlc.LABEL, f"Label column '{label_column_name}' must have role tlc.LABEL={tlc.LABEL}."

        # Check for data in columns
        assert len(self.table) > 0, f"Table {self.root.to_str()} has no rows."
        first_row = self.table.table_rows[0]
        assert isinstance(first_row[image_column_name], str), f"First value in image column '{image_column_name}' in table {self.root.to_str()} is not a string."
        assert isinstance(first_row[label_column_name], int), f"First value in label column '{label_column_name}' in table {self.root.to_str()} is not an integer."

    def verify_images(self):
        """ Verify all images in the dataset."""

        verified_marker_url = self.table.url / "cache.yolo"

        # If the marker exists, we can skip verification
        if verified_marker_url.exists():
            LOGGER.info(f"{self.prefix}Images in {self.root.to_str()} already verified, skipping scan.")
            return self.samples

        desc = f"{self.prefix}Scanning images in {self.root.to_str()}..."
        # Run scan if the marker does not exist
        nf, nc, msgs, samples, example_ids = 0, 0, [], [], []
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(enumerate(results), desc=desc, total=len(self.samples))
            for i, (sample, nf_f, nc_f, msg) in pbar:
                if nf_f:
                    example_ids.append(self.example_ids[i])
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))

        # If no problems are found, create the marker
        if nc == 0:
            LOGGER.info(f"{self.prefix}All images in {self.root.to_str()} are verified. Writing marker file to {verified_marker_url.to_str()} to skip future verification.")
            verified_marker_url.write(
                content=json.dumps({"verified": True}),
                if_exists="raise", # Should not get here if already exists
            )

        self._example_ids = example_ids
        return samples