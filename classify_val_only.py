import tlc

from ultralytics.utils.tlc import TLCYOLO, Settings

model = TLCYOLO("yolov8n-cls.pt")

settings = Settings(
    run_description="Collect mode for cifar10!",
    image_embeddings_dim=2,
)

tables = {
    "train": tlc.Table.from_url("/home/fredmell/yolov5-tutorial/3LC/cifar10-YOLOv8/datasets/cifar10-train/tables/original"),
    "test": tlc.Table.from_url("/home/fredmell/yolov5-tutorial/3LC/cifar10-YOLOv8/datasets/cifar10-test/tables/original"),
}

results = model.collect(tables=tables, batch=32, imgsz=640, device=0, workers=0, settings=settings)
