# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
import cv2
import numpy as np
import pathlib
import tlc

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.tlc.engine.validator import TLCValidatorMixin
from ultralytics.utils.tlc.detect.utils import build_tlc_yolo_dataset, yolo_predicted_bounding_box_schema

class TLCSegmentationValidator(TLCValidatorMixin, SegmentationValidator):
    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader with given parameters."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def build_dataset(self, table, mode="val", batch=None):
        return build_tlc_yolo_dataset(self.args, table, batch, self.data, mode=mode, stride=self.stride, settings=self._settings)
    
    def _get_metrics_schemas(self):
        map = {float(cat_idx + 1): tlc.MapElement(cat_val) for cat_idx, cat_val in self.data["names"].items()}
        return {
            tlc.PREDICTED_MASK: tlc.SegmentationMaskUrlStringValue(map=map),
        }
    
    def _compute_3lc_metrics(self, preds, batch):
        segmentation_urls = []
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            
            conf = predn[:, 4]
            pred_cls = predn[:, 5]

            pred_masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])
            scaled_masks = ops.scale_image(
                pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                pbatch["ori_shape"],
                ratio_pad=batch["ratio_pad"][si],
            )

            shape = batch["ori_shape"][si]
            h, w = shape
            output_segmentation = np.zeros(shape=shape, dtype=np.uint8)
            for j in range(len(conf)):
                if conf[j] > 0.1: # TODO: use setting
                    mh, mw = scaled_masks[...,j].shape
                    assert mh == h and mw == w, "Mask shape should now be the same as image shape"
                    mask = scaled_masks[...,j].astype(bool)
                    output_segmentation[mask] = pred_cls[j].item() + 1 # Add one to leave 0 as background
        
            # write to desired location and write the location
            epoch_str = f"epoch_{self._epoch}" if self._epoch is not None else "after"
            image_path = batch['im_file'][si]
            bulk_data_url = self._run.bulk_data_url
            segmentation_url = bulk_data_url / epoch_str / pathlib.Path(image_path).with_suffix(".png").name # Same name as image
            segmentation_url.make_parents(exist_ok=True)

            cv2.imwrite(segmentation_url.to_absolute().to_str(), output_segmentation)
            segmentation_urls.append(segmentation_url.to_str())

        return {
            tlc.PREDICTED_MASK: segmentation_urls,
        }

    def _infer_batch_size(self, preds, batch) -> int:
        return len(batch['im_file'])