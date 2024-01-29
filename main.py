print("Running")

from ultralytics import YOLO  # noqa: E402

model = YOLO('yolov8n.pt')  # initialize

results = model.train(data='coco128.yaml', model='yolov8x.pt', epochs=2, batch=32, imgsz=320, workers=0)