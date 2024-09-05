# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer

from ultralytics.utils import LOGGER
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME, DETECTION_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.detect.utils import build_tlc_yolo_dataset, tlc_check_det_dataset
from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator
from ultralytics.utils.tlc.engine.trainer import TLCTrainerMixin

from ultralytics.utils.torch_utils import de_parallel

class TLCDetectionTrainer(TLCTrainerMixin, DetectionTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = DETECTION_LABEL_COLUMN_NAME

    """Trainer class for YOLOv8 object detection with 3LC"""
    def get_dataset(self):
        # Parse yaml and create tables
        self.data = tlc_check_det_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            splits=("train", "val", "test"),
            project_name=self._settings.project_name,
        )
        # Need at least a 'val' or 'test' split for validation during training
        if not any(key in self.data for key in ("val", "test")):
            # Check for 'validation' to match ultralytics/utils.py equivalent check_det_dataset
            if "validation" in self.data:
                LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
                self.data["val"] = self.data.pop("validation")  # replace 'validation' key with 'val' key
            else:
                raise ValueError("A 'val' or 'test' split is required for detection training.")
        return self.data["train"], self.data.get("val") or self.data.get("test")
    
    def build_dataset(self, table, mode="train", batch=None):
        # Dataset object for training / validation
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_tlc_yolo_dataset(
            self.args, table, batch, self.data, mode=mode, rect=mode == "val", stride=gs, settings=self._settings
        )

    def get_validator(self, dataloader=None):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if not dataloader:
            dataloader = self.test_loader

        return TLCDetectionValidator(
            dataloader,
            save_dir=self.save_dir,
            args=self.args,
            run=self._run,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            settings=self._settings,
        )
    
    def _process_metrics(self, metrics):
        return {
            metric.strip("(B)").replace("metrics", "val").replace("/", "_"): value
            for metric, value in metrics.items()
        }