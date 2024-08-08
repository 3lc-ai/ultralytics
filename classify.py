import tlc

from ultralytics.utils.tlc.detect.model import TLCYOLO  # noqa: E402
from ultralytics.utils.tlc.detect.settings import Settings  # noqa: E402

# Load a model
model = TLCYOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

settings = Settings(
    sampling_weights=True,
    exclude_zero_weight_training=True,
    exclude_zero_weight_collection=True,
)

# Train the model using tables directly

results = model.train(
    data="cifar10",
    epochs=1,
    imgsz=64,
    batch=64,
    settings=settings
)