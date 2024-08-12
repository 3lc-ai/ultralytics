from ultralytics.models.yolo.model import YOLO

from ultralytics.nn.tasks import ClassificationModel, DetectionModel
from ultralytics.utils.tlc.classify import TLCClassificationTrainer, TLCClassificationValidator
from ultralytics.utils.tlc.detect import TLCDetectionTrainer, TLCDetectionValidator

class TLCYOLO(YOLO):
    """ YOLO (You Only Look Once) object detection model with 3LC integration. """

    @property
    def task_map(self):
        """ Map head to 3LC model, trainer, validator, and predictor classes. """
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": TLCDetectionTrainer,
                "validator": TLCDetectionValidator,
            },
            "classify": {
                "model": ClassificationModel,
                "trainer": TLCClassificationTrainer,
                "validator": TLCClassificationValidator,
            },
        }
