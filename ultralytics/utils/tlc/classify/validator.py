import tlc
import torch

from ultralytics.models import yolo
from ultralytics.data import build_dataloader
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME, CLASSIFY_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.classify.dataset import TLCClassificationDataset
from ultralytics.utils.tlc.engine.validator import TLCValidator


class TLCClassificationValidator(TLCValidator, yolo.classify.ClassificationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = CLASSIFY_LABEL_COLUMN_NAME

    def build_dataset(self, table):
        return TLCClassificationDataset(
            table=table,
            args=self.args,
            augment=False,
            prefix=self.args.split,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            exclude_zero_weight=self._settings.exclude_zero_weight_collection,
            settings=self._settings
        )
    
    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def _get_metrics_schemas(self):
        class_names=[value['internal_name'] for value in self.dataloader.dataset.table.get_value_map(self._label_column_name).values()]
        column_schemas = {
            "loss": tlc.Float32Value(),
            "predicted": tlc.CategoricalLabelSchema(class_names=class_names, display_name="Predicted"), # TODO: Improve the description?
            "confidence": tlc.Float32Value(value_min=0.0, value_max=1.0),
            "top1_accuracy": tlc.Float32Value(),
        }
        if len(class_names) > 5:
            column_schemas["top5_accuracy"] = tlc.Float32Value()
        return column_schemas

    def _compute_3lc_metrics(self, preds, batch):
        """ Update 3LC metrics """
        confidence, predicted = preds.max(dim=1)

        batch_metrics = {
            "loss": torch.nn.functional.nll_loss(torch.log(preds), batch["cls"], reduction="none"), #nll since preds are normalized
            "predicted": predicted,
            "confidence": confidence,
            "top1_accuracy": (torch.argmax(preds, dim=1) == batch["cls"]).to(torch.float32),
        }

        if len(self.dataloader.dataset.table.get_value_map(self._label_column_name)) > 5:
            _, top5_pred = torch.topk(preds, 5, dim=1)
            labels_expanded = batch["cls"].view(-1, 1).expand_as(top5_pred)
            top5_correct = torch.any(top5_pred == labels_expanded, dim=1)
            batch_metrics["top5_accuracy"] = top5_correct.to(torch.float32)

        return batch_metrics
    
    def _verify_model_data_compatibility(self, names):
        class_names={
            float(i): value['internal_name']
            for i, value in enumerate(self.dataloader.dataset.table.get_value_map(self._label_column_name).values())
        }
        if names != class_names:
            raise ValueError(
                "The model and data are incompatible. " 
                "The model was trained with different classes than the data on which val() has been called."
            )
