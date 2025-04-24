import tlc

from ultralytics.utils.tlc import TLCYOLO, Settings

if __name__ == "__main__":
    # train_url = "/Users/frederik/Library/Application Support/3LC/projects/segmentation-dev/datasets/coco128-seg-val/tables/initial"
    # val_url = "/Users/frederik/Library/Application Support/3LC/projects/segmentation-dev/datasets/coco128-seg-train/tables/initial"

    # train_table = tlc.Table.from_url(train_url)
    # val_table = tlc.Table.from_url(val_url)

    model = TLCYOLO("yolo11n-seg.pt")

    # tables = {
    #     "train": train_table,
    #     "val": val_table,
    # }
    model.train(data="coco8-seg.yaml", epochs=1, imgsz=640, settings=Settings(conf_thres=0.1), agnostic_nms=True)
