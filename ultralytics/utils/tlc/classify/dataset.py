import tlc

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

        self.samples = []
        self.example_ids = []

        iterator = self._get_enumerated_table_rows(exclude_zero_weight=exclude_zero_weight)
        for example_id, row in iterator:
            self.example_ids.append(example_id)
            self.samples.append((Path(row[image_column_name]), row[label_column_name]))

        # Initialize attributes
        self._init_attributes(args, augment, prefix)

        # Call mixin
        super().__init__(**kwargs)

        self._indices = np.arange(len(self.example_ids))
        assert len(self._indices) == len(self.samples)

    def verify_images(self):
        """ Verify all images in the dataset."""
        desc = f"{self.prefix}Scanning images in {self.root.to_str()}..."

        # TODO: Consider saving cache next to table (see parent)

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

        self._example_ids = example_ids
        return samples