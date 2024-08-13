import tlc

from ultralytics.utils.tlc import TLCYOLO, Settings

# Load a model
model = TLCYOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

settings = Settings(
    sampling_weights=True,
    exclude_zero_weight_training=True,
    exclude_zero_weight_collection=True,
    collection_epoch_start=0,
    collection_epoch_interval=2,
    run_description="My description",
    run_name="My run name",
    # image_embeddings_dim=3,
)

# Train the model using tables directly
results = model.train(
    data="cifar10",
    epochs=10,
    imgsz=64,
    batch=64,
    settings=settings
)