from ultralytics.utils.tlc import TLCYOLO

from ultralytics.utils.tlc.detect.settings import Settings

model = TLCYOLO("yolov8s.pt")  # initialize

# Set 3LC specific settings
settings = Settings(
    image_embeddings_dim=3,
    # collection_epoch_start=1,
    # collection_epoch_interval=2,
    conf_thres=0.2,
    run_description="3D embeddings!",
)


# Run training
results = model.train(
    data="coco128.yaml",
    device=0,
    epochs=10,
    batch=32,
    imgsz=320,
    workers=0,
    settings=settings,
)
