import numpy as np
import tlc
import torch
import weakref
import scipy

from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops
# from ultralytics.utils.tlc.engine.validator import TLCValidatorMixin
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME
from ultralytics.utils.tlc.detect.utils import build_tlc_yolo_dataset
from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator
from ultralytics.utils.tlc.segment.utils import compute_mask_iou, tlc_check_seg_dataset

SEGMENTATION_LABEL_COLUMN_NAME = "segmentation_label"

class TLCSegmentationValidator(TLCDetectionValidator, SegmentationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = SEGMENTATION_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return tlc_check_seg_dataset(*args, **kwargs)

    def _get_metrics_schemas(self):
        #TODO: Ensure class  mapping is the same as in input table
        instance_properties_structure = {
            tlc.LABEL: tlc.CategoricalLabel(name=tlc.LABEL, classes=self.data["names"]),
            tlc.CONFIDENCE: tlc.Float(name=tlc.CONFIDENCE, number_role=tlc.NUMBER_ROLE_CONFIDENCE),
            "matched_gt_iou": tlc.Float(name="matched_gt_iou", number_role=tlc.NUMBER_ROLE_IOU),
            "matched_gt_id": tlc.Int(name="matched_gt_id", number_role="instance_index"),
        }

        segment_sample_type = tlc.InstanceSegmentationMasks(
            name="predicted_segmentations",
            instance_properties_structure=instance_properties_structure,
            is_prediction=True,
        )

        return {"predicted_segmentations": segment_sample_type.schema}

    def _compute_3lc_metrics(self, preds, batch):
        predicted_batch_segmentations = []

        # Reimplements of SegmentationValidator, but with control over mask processing
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
                        "matched_gt_iou": [],
                        "matched_gt_id": [],
                    },
                    "masks": [],
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

                pred_class = pred_cls.cpu().numpy().astype(np.int32)
                true_class = cls.cpu().numpy().astype(np.int32)
                correct_class = true_class[:, None] == pred_class
                iou = mask_iou.numpy() * correct_class

                label_idx, prediction_idx = scipy.optimize.linear_sum_assignment(iou, maximize=True)
                
                # Initialize all predictions as unmatched (-1)
                matched_gt_id = np.full(len(pred_cls), -1)
                matched_gt_iou = np.zeros(len(pred_cls))
                
                # Only update the matched predictions
                matched_gt_id[prediction_idx] = label_idx
                matched_gt_iou[prediction_idx] = iou[label_idx, prediction_idx]

                # Filter out matches with zero IoU first
                valid = matched_gt_iou > 0
                matched_gt_id[np.logical_not(valid)] = -1

                # Map back to original instance order
                instance_mapping = batch["instance_mapping"][si]
                reverse_instance_mapping = {v: i for i, v in instance_mapping.items()}
                # Only map the valid matches (where matched_gt_id != -1)
                matched_gt_id[valid] = np.array([reverse_instance_mapping[index] for index in matched_gt_id[valid]])

                # TODO: Map back classes to original classes (see detection validator)

                assert len(matched_gt_id) == len(pred_cls), "Number of matched ground truth ids does not match number of predictions"
                # Assert matched_gt_ids greater than -1 are unique
                assert len(np.unique(matched_gt_id[matched_gt_id > -1])) == len(matched_gt_id[matched_gt_id > -1]), "Matched ground truth ids are not unique"
            else:
                matched_gt_iou = torch.zeros(len(pred_cls))
                matched_gt_id = torch.full_like(matched_gt_iou, -1)

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
                    "matched_gt_iou": matched_gt_iou.tolist(),
                    "matched_gt_id": matched_gt_id.tolist(),
                },
                "masks": result_masks,
            }

            predicted_batch_segmentations.append(predicted_instances)

        return {"predicted_segmentations": predicted_batch_segmentations}
            

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
