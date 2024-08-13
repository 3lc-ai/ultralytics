from ultralytics.utils.tlc.detect.model import TLCYOLO
from ultralytics.utils.tlc.detect.settings import Settings
from ultralytics.utils.tlc.detect.utils import reduce_all_embeddings

splits = ("train", "val")
data="coco128.yaml"

model = TLCYOLO("yolov8l.pt")

settings = Settings(
    image_embeddings_dim=2,
    conf_thres=0.2,
)

model.collect(data=data, splits=splits, settings=settings, batch=32, imgsz=320, device=0, workers=0)