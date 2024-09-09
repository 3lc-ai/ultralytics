# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license

from copy import copy

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils.tlc.constants import SEGMENTATION_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.detect.trainer import TLCDetectionTrainer
from ultralytics.utils.tlc.segment.validator import TLCSegmentationValidator

class TLCSegmentationTrainer(SegmentationTrainer, TLCDetectionTrainer):
    _default_label_column_name = SEGMENTATION_LABEL_COLUMN_NAME

    def get_validator(self, dataloader=None):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"

        if not dataloader:
            dataloader = self.test_loader

        return TLCSegmentationValidator(
            dataloader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            run=self._run,
            settings=self._settings,
        )