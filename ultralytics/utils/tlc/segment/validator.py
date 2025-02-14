# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license

import numpy as np
import tlc
import torch
from tlc.client.data_format import InstanceSegmentationDict

from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME
from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator
from ultralytics.utils.tlc.segment.utils import compute_mask_iou, tlc_check_seg_dataset

SEGMENTATION_LABEL_COLUMN_NAME = "segmentation_label" # TODO: Make a constant and use it?

class TLCSegmentationValidator(TLCDetectionValidator, SegmentationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = SEGMENTATION_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return tlc_check_seg_dataset(*args, **kwargs)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        #TODO: Ensure class  mapping is the same as in input table
        instance_properties_structure = {
            tlc.LABEL: tlc.CategoricalLabel(name=tlc.LABEL, classes=self.data["names"]),
            tlc.CONFIDENCE: tlc.Float(name=tlc.CONFIDENCE, number_role=tlc.NUMBER_ROLE_CONFIDENCE),
        }

        segment_sample_type = tlc.InstanceSegmentationMasks(
            name="predicted_segmentations",
            instance_properties_structure=instance_properties_structure,
            is_prediction=True,
        )

        return {"predicted_segmentations": segment_sample_type.schema}

    def _compute_3lc_metrics(self, preds, batch) -> list[dict[str, InstanceSegmentationDict]]:
        """Compute 3LC metrics for instance segmentation.
        
        :param preds: Predictions returned by YOLO segmentation model.
        :param batch: Batch of data presented to the YOLO segmentation model.
        :returns: Metrics dict with predicted instance data for each sample in a batch.
        """
        predicted_batch_segmentations = []

        # Reimplements SegmentationValidator, but with control over mask processing
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            gt_masks = pbatch.pop("masks")

            conf = pred[:, 4]
            pred_cls = pred[:, 5]

            # Filter out predictions first
            keep_indices = conf >= self._settings.conf_thres

            # Handle case where no predictions are kept
            if not torch.any(keep_indices):
                predicted_instances = {
                    "image_height": pbatch["ori_shape"][0],
                    "image_width": pbatch["ori_shape"][1],
                    "instance_properties": {
                        tlc.LABEL: [],
                        tlc.CONFIDENCE: [],
                    },
                    tlc.GT_IOUS: [],
                    tlc.MASKS: [],
                }
                predicted_batch_segmentations.append(predicted_instances)
                continue
            
            pred_cls = pred_cls[keep_indices]
            conf = conf[keep_indices]
            pred = pred.detach().clone()[keep_indices]

            # Native upsampling to bounding boxes
            prev_process = self.process
            self.process = ops.process_mask_native
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            self.process = prev_process

            # Only match when there are labels
            if len(cls) > 0:
                mask_iou = compute_mask_iou(cls, pred_masks=pred_masks, gt_masks=gt_masks, overlap=True)

                # Reorder using instance mapping before transposing and converting to list
                instance_mapping = batch["instance_mapping"][si]
                indices = list(instance_mapping.keys())
                mask_iou = mask_iou[indices, :].transpose(1, 0).tolist()
            else:
                mask_iou = [[] for _ in range(len(pred_cls))]

            # Scale masks to image size and handle padding
            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)

            scaled_masks = ops.scale_image(
                pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                pbatch["ori_shape"],
                ratio_pad=batch["ratio_pad"][si],
            )

            result_masks = np.asfortranarray(scaled_masks.astype(np.uint8))

            predicted_instances = {
                "image_height": pbatch["ori_shape"][0],
                "image_width": pbatch["ori_shape"][1],
                "instance_properties": {
                    tlc.LABEL: pred_cls.tolist(),
                    tlc.CONFIDENCE: conf.tolist(),
                },
                tlc.GT_IOUS: mask_iou, 
                tlc.MASKS: result_masks,
            }

            predicted_batch_segmentations.append(predicted_instances)

        return {"predicted_segmentations": predicted_batch_segmentations}
