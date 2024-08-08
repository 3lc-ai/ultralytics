import tlc

from ultralytics.utils.tlc import TLCYOLO, Settings

model = TLCYOLO("yolov8n-cls.pt")

settings = Settings()

# data + split should be okay
# table should be okay
table = tlc.Table.from_url("/home/fredmell/yolov5-tutorial/3LC/cifar10-YOLOv8/datasets/cifar10-test/tables/original")

results = model.val(table=table, batch=32, imgsz=640, device=0, workers=0, settings=settings)
