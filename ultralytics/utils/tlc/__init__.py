# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
import time
from ultralytics.utils import LOGGER, colorstr

_msg = (
    f"{colorstr('red', 'bold','WARNING:')} "
    "The Ultralytics 3LC integration in this repository is no longer maintained. "
    "Please use the 3LC integration from https://github.com/3lc-ai/3lc-ultralytics, "
    "which can be installed from PyPI with 'pip install 3lc-ultralytics'."
)
LOGGER.warning(_msg)

time.sleep(2)

from .classify import TLCClassificationTrainer, TLCClassificationValidator
from .detect import TLCDetectionTrainer, TLCDetectionValidator
from .segment import TLCSegmentationTrainer, TLCSegmentationValidator
from .settings import Settings
from .engine.model import TLCYOLO

__all__ = [
    "Settings",
    "TLCYOLO",
    "TLCClassificationTrainer",
    "TLCClassificationValidator",
    "TLCDetectionTrainer",
    "TLCDetectionValidator",
    "TLCSegmentationTrainer",
    "TLCSegmentationValidator",
]
