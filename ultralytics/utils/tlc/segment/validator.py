# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
import cv2
import io
import numpy as np
import pathlib
import tlc
import torch
import weakref

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.tlc.engine.validator import TLCValidatorMixin
from ultralytics.utils.tlc.detect.utils import build_tlc_yolo_dataset, yolo_predicted_bounding_box_schema, construct_bbox_struct

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
            tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(
                {idx+1: name for idx, name in self.data["names"].items()}
            ),
        }
    
    def _compute_3lc_metrics(self, preds, batch):
        segmentation_urls = []
        bb_list = []
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            img = batch["img"][si]
            original_image = batch["im_file"][si]
            original_image_shape = batch["ori_shape"][si]
            pred[:, :4] = ops.scale_boxes(img.shape[1:], pred[:, :4], original_image_shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], original_image_shape[:2])  # HWC
            
            from ultralytics.engine.results import Masks

            result_masks = Masks(masks, original_image_shape)
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            
            conf = predn[:, 4]
            pred_cls = predn[:, 5]

            # pred_masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])
            # scaled_masks = ops.scale_image(
            #     pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
            #     pbatch["ori_shape"],
            #     ratio_pad=batch["ratio_pad"][si],
            # )

            shape = batch["ori_shape"][si]
            h, w = shape

            pred_boxes_xyxy = predn[:, :4]
            pred_boxes_xywhn = ops.xyxy2xywhn(pred_boxes_xyxy, w=w, h=h)
            
            bbs = []
            output_segmentation = np.zeros(shape=shape, dtype=np.uint8)
            for j in range(len(conf)):
                if conf[j] > 0.1: # TODO: use setting
                    category = pred_cls[j].item() + 1 # Add one to leave 0 as background
                    # Box
                    bbs.append(
                        {
                            "score": conf[j].item(),
                            "category_id": int(category), # Add one to leave 0 as background
                            "bbox": pred_boxes_xywhn[j].tolist(),
                            "iou": 0.0, # TODO: compute IoU, what about mask IoU?
                        }
                    )

                    # Segmentation
                    # output_segmentation = cv2.polylines(
                    #     output_segmentation,
                    #     pts=[]
                    # )
                    # mh, mw = scaled_masks[...,j].shape
                    # assert mh == h and mw == w, "Mask shape should now be the same as image shape"
                    # mask = scaled_masks[...,j].astype(bool)
                    # output_segmentation[mask] = category # Add one to leave 0 as background
        
            # add bbs
            bb_list.append(construct_bbox_struct(bbs, image_width=w, image_height=h))

            # write seg to desired location and write the location
            
            epoch_str = f"epoch_{self._epoch}" if self._epoch is not None else "after"
            image_path = batch['im_file'][si]
            bulk_data_url = self._run.bulk_data_url
            segmentation_url = bulk_data_url / epoch_str / pathlib.Path(image_path).with_suffix(".png").name # Same name as image
            segmentation_url.make_parents(exist_ok=True)

            success, buffer = cv2.imencode(".png", output_segmentation)
            with io.BytesIO(buffer) as io_buffer:
                segmentation_url.write(io_buffer.getvalue())

            segmentation_url_relative = segmentation_url.to_relative(self._metrics_writer.url)
            segmentation_urls.append(segmentation_url_relative.to_str())

        return {
            tlc.PREDICTED_MASK: segmentation_urls,
            tlc.PREDICTED_BOUNDING_BOXES: bb_list,
        }

    def _add_embeddings_hook(self, model) -> int:
        if hasattr(model.model, "model"):
            model = model.model

        # Find index of the SPPF layer
        sppf_index = next((i for i, m in enumerate(model.model) if "SPPF" in m.type), -1)

        if sppf_index == -1:
            raise ValueError("No SPPF layer found in model, cannot collect embeddings.")
        
        weak_self = weakref.ref(self) # Avoid circular reference (self <-> hook_fn)
        def hook_fn(module, input, output):
            # Store embeddings
            self_ref = weak_self()
            flattened_output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
            embeddings = flattened_output.detach().cpu().numpy()
            self_ref.embeddings = embeddings

        # Add forward hook to collect embeddings
        for i, module in enumerate(model.model):
            if i == sppf_index:
                self._hook_handles.append(module.register_forward_hook(hook_fn))

        activation_size = model.model[sppf_index]._modules['cv2']._modules['conv'].out_channels
        return activation_size

    def _infer_batch_size(self, preds, batch) -> int:
        return len(batch['im_file'])