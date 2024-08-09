# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
import ultralytics

from ultralytics.utils.tlc.detect.utils import get_names_from_yolo_table, tlc_check_dataset
from ultralytics.utils.tlc.engine.model import TLCYOLO


def check_det_dataset(data: str):
    """Check if the dataset is compatible with the 3LC."""
    tables = tlc_check_dataset(data)
    names = get_names_from_yolo_table(tables["train"])
    return {
        "train": tables["train"],
        "val": tables["val"],
        "nc": len(names),
        "names": names, }


ultralytics.engine.validator.check_det_dataset = check_det_dataset

TLCYOLO = TLCYOLO