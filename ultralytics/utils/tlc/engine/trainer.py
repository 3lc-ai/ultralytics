import tlc

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.tlc.detect.settings import Settings
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import model_info_for_loggers, strip_optimizer

# TODO:
# Schema check? - Gudbrand
# DDP training
# Adapt to object detection task in case of any problems
  
class TLCTrainer(BaseTrainer):
    """ A class extending the BaseTrainer class for training Ultralytics YOLO models with 3LC,
    which implements common 3LC-specific behavior across tasks. Use as one of two base classes for
    task-specific trainers.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        LOGGER.info("Using 3LC Trainer 🌟")
        self._settings = Settings() if 'settings' not in overrides else overrides.pop('settings')
        self._settings.verify(training=True)

        assert 'data' in overrides or 'tables' in overrides, "You must provide either a data path or tables to train with 3LC."
        self._tables = overrides.pop('tables', None)

        # Column names
        self._image_column_name = overrides.pop("image_column_name", self._default_image_column_name)
        self._label_column_name = overrides.pop("label_column_name", self._default_label_column_name)

        super().__init__(cfg, overrides, _callbacks)

        self._metrics_collection_epochs = set(self._settings.get_metrics_collection_epochs(self.epochs))
        self._train_validator = None

        # Create a 3LC run
        if RANK in {-1, 0}:
            description = self._settings.run_description if self._settings.run_description else "Created with 3LC Ultralytics Integration"
            project_name = self._settings.project_name if self._settings.project_name else self.data["train"].project_name
            self._run = tlc.init(
                project_name=project_name,
                description=description,
            )

        # Log parameters to 3LC
        self._log_3lc_parameters()

        self.add_callback("on_train_epoch_start", resample_indices)
        
    def _log_3lc_parameters(self):
        if "val" in self.data:
            val_url = str(self.data["val"].url)
        else:
            val_url = str(self.data["test"].url)

        parameters = {
            **vars(self.args), # YOLO arguments
            "3LC/train_url": str(self.data.get("train").url), # 3LC table used for training
            "3LC/val_url": val_url, # 3LC table used for validation
            **{f"3LC/{k}": v for k, v in vars(self._settings).items()}, # 3LC settings
        }
        self._run.set_parameters(parameters)

    def get_dataset(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    @property
    def train_validator(self):
        if RANK in {-1, 0}:
            if not self._train_validator:
                train_validator_dataloader = self.get_dataloader(
                    self.trainset,
                    batch_size=self.batch_size if self.args.task == "obb" else self.batch_size * 2,
                    rank=-1, 
                    mode="val"
                )
                self._train_validator = self.get_validator(
                    dataloader=train_validator_dataloader
                )
            return self._train_validator

    def validate(self):
        """ Perform validation with 3LC metrics collection, also on the training data, if applicable."""
        # Validate on the training set
        if not self._settings.collection_disable and not self._settings.collection_val_only and self.epoch in self._metrics_collection_epochs:
            self.train_validator(trainer=self)
        
        # Validate on the validation/test set like usual
        return super().validate()
    
    def final_eval(self):
        self.train_validator._final_validation = True
        # Set epoch on validator - required when final validation is called without prior mc during training
        self.train_validator._epoch = self.epoch
        self.validator._final_validation = True

        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.train_validator(model=f)
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

        if self._settings.image_embeddings_dim > 0:
            self._run.reduce_embeddings_by_foreign_table_url(
                foreign_table_url=self.data["train"].url,
                method=self._settings.image_embeddings_reducer,
                n_components=3,
            )
        self._run.set_status_completed()

    def save_metrics(self, metrics):
        # Log model info only after first epoch
        # if self.epoch == 0:
        #     self._run.add_output_value(model_info_for_loggers(self))
        
        # Log aggregate metrics after every epoch
        self._run.add_output_value({"epoch": self.epoch, **self.metrics})
        
        super().save_metrics(metrics=metrics)

# CALLBACKS ##############################################################################################################
def resample_indices(trainer):
    # TODO: DDP - verify called by all processes
    if trainer._settings.sampling_weights:
        trainer.train_loader.dataset.resample_indices()