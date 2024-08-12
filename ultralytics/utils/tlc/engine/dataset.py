import tlc
import numpy as np

# Responsible for any generic 3LC dataset handling, such as using sampling weights
# Assume there is an attribute self.table that is a tlc.Table, and self._example_ids
class TLCDatasetMixin:
    def _post_init(self, sampling_weights=False):
        # Checks
        if sampling_weights and tlc.SAMPLE_WEIGHT not in self.table.table_rows[0]:
            raise ValueError("Cannot use sampling weights with no sample weight column in the table.")

        assert hasattr(self, "table") and isinstance(self.table, tlc.Table), "TLCDatasetMixin requires an attribute `table` which is a tlc.Table."
        # Assume instance has self._indices (live sampled indices of the dataset)
        # Assume instance has self._example_ids (index -> example_id mapping)

        if not hasattr(self, "_indices"):
            self._indices = np.arange(len(self.example_ids))

        sample_weights = [
            self.table.table_rows[example_id][tlc.SAMPLE_WEIGHT]
            for example_id in self.example_ids
        ]
        self._sample_probabilities = np.array(sample_weights) / np.sum(sample_weights)

    def resample_indices(self):
        # Sample from available indices
        self._indices[:] = np.random.choice(self.example_ids, len(self.example_ids), p=self._sample_probabilities)

    def __getitem__(self, index):
        i = self._indices[index]
        return super().__getitem__(i)
    
    def __len__(self):
        return len(self._indices)

    def _get_enumerated_table_rows(self, exclude_zero_weight):
        if exclude_zero_weight and tlc.SAMPLE_WEIGHT not in self.table.table_rows[0]:
            raise ValueError("Cannot exclude zero weight samples with no sample weight column in the table.")

        if exclude_zero_weight:
            return ((i, row) for i, row in enumerate(self.table.table_rows) if row[tlc.SAMPLE_WEIGHT] > 0)
        else:
            return enumerate(self.table.table_rows)