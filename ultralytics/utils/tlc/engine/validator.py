import tlc
import weakref

from torch import nn

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import colorstr
from ultralytics.utils.tlc.detect.settings import Settings
from ultralytics.utils.tlc.detect.utils import training_phase_schema
from ultralytics.utils.tlc.classify.utils import tlc_check_cls_dataset

def execute_when_collecting(method):
    def wrapper(self, *args, **kwargs):
        if self._should_collect:
            return method(self, *args, **kwargs)
    return wrapper

class TLCValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, run=None, image_column_name=None, label_column_name=None, settings=None):

        # Called by trainer (Get run and settings from trainer)
        if run is not None:
            self._run = run
            self._settings = settings
            self._image_column_name = image_column_name
            self._label_column_name = label_column_name
            self._training = True

        # Called directly (Create a run and get settings directly)
        else:
            if run in args:
                self._run = args.pop("run")
            else:
                self._run = None # Create run
            self._settings = args.pop("settings", Settings())
            self._image_column_name = args.pop("image_column_name", self._default_image_column_name)
            self._label_column_name = args.pop("label_column_name", self._default_label_column_name)
            self._table = args.pop("table", None)
            self._training = False

        # State
        self._activation_size = None
        self._epoch = None
        self._should_collect = None
        self._seen = None
        self._final_validation = False
        self._hook_handles = []

        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        if not self._training:
            self.data = tlc_check_cls_dataset(
                self.args.data,
                {self.args.split: self._table} if self._table is not None else None,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
            )

        if self._run is None:
            first_split = list(self.data.keys())[0]
            project_name = self._settings.project_name if self._settings.project_name else self.data[first_split].project_name
            self._run = tlc.init(
                project_name=project_name,
                description=self._settings.run_description if self._settings.run_description else "Created with 3LC Ultralytics Integration",
                run_name=self._settings.run_name,
            )

    def __call__(self, trainer=None, model=None):
        self._epoch = trainer.epoch if trainer is not None else self._epoch
        
        if trainer:
            self._should_collect = not self._settings.collection_disable and self._epoch in trainer._metrics_collection_epochs
        else:
            self._should_collect = True

        # self._pre_validation()

        # Call parent to perform the validation
        out = super().__call__(trainer, model)

        self._post_validation()

        return out
    
    def get_desc(self):
        desc = super().get_desc()

        split = self.dataloader.dataset.display_name.split("-")[-1]
        initial_spaces = len(desc) - len(desc.lstrip())
        split_centered = split.center(initial_spaces)
        split_str = f"{colorstr(split_centered)}"
        desc = split_str + desc[len(split_centered):]

        return desc
    
    def init_metrics(self, model):
        super().init_metrics(model)

        self._verify_model_data_compatibility(model.names)
        self._add_embeddings_hook(model)
        self._pre_validation()

    def _verify_model_data_compatibility(self, names):
        raise NotImplementedError("Subclasses must implement this method.")

    @execute_when_collecting
    def _add_embeddings_hook(self, model):
        """ Add a hook to extract embeddings from the model, and infer the activation size """
        if self._settings.image_embeddings_dim > 0:
            # Find index of the linear layer
            has_linear_layer = False
            for index, module in enumerate(model.modules()):
                if isinstance(module, nn.Linear):
                    has_linear_layer = True
                    self._activation_size = module.in_features
                    break

            if not has_linear_layer:
                raise ValueError("No linear layer found in model, cannot collect embeddings.")

            weak_self = weakref.ref(self) # Avoid circular reference (self <-> hook_fn)
            def hook_fn(module, input, output):
                # Store embeddings
                self_ref = weak_self()
                embeddings = output.detach().cpu().numpy()
                self_ref.embeddings = embeddings

            # Add forward hook to collect embeddings
            for i, module in enumerate(model.modules()):
                if i == index - 1:
                    self._hook_handles.append(module.register_forward_hook(hook_fn))
    
    def update_metrics(self, preds, batch):
        """ Collect 3LC metrics """
        self._update_metrics(preds, batch)

        # Let parent collect its own metrics
        super().update_metrics(preds, batch)

    @execute_when_collecting
    def _update_metrics(self, preds, batch):
        batch_size = preds.size(0)
        example_indices = list(range(self._seen, self._seen + batch_size))
        example_ids = [self.dataloader.dataset.example_ids[i] for i in example_indices]

        batch_metrics = {
            tlc.EXAMPLE_ID: example_ids,
            **self._compute_3lc_metrics(preds, batch)
        }

        if self._settings.image_embeddings_dim > 0:
            batch_metrics["embeddings"] = self.embeddings

        if self._epoch is not None:
            batch_metrics[tlc.EPOCH] = [self._epoch] * batch_size
            training_phase = 1 if self._final_validation else 0
            batch_metrics["Training Phase"] = [training_phase] * batch_size

        self._metrics_writer.add_batch(batch_metrics)
        self._seen += batch_size

    @execute_when_collecting
    def _pre_validation(self):
        column_schemas = {}
        column_schemas.update(self._get_metrics_schemas()) # Add task-specific metrics schema

        if self._settings.image_embeddings_dim > 0:
            column_schemas["embeddings"] = tlc.Schema(
                'Embedding',
                'Large NN embedding',
                writable=False,
                computable=False,
                value=tlc.Float32Value(number_role=tlc.NUMBER_ROLE_NN_EMBEDDING),
                size0=tlc.DimensionNumericValue(
                    value_min=self._activation_size,
                    value_max=self._activation_size,
                    enforce_min=True,
                    enforce_max=True
                )
            )

        if self._epoch is not None:
            column_schemas["Training Phase"] = training_phase_schema()

        self._run.set_status_collecting()

        self._metrics_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url, # TODO: Generalize
            foreign_table_display_name=self.dataloader.dataset.display_name,
            column_schemas=column_schemas
        )

        self._seen = 0

    @execute_when_collecting
    def _post_validation(self):
        self._metrics_writer.finalize()
        metrics_infos = self._metrics_writer.get_written_metrics_infos()
        self._run.update_metrics(metrics_infos)

        self._run.set_status_running()

        # Remove hook handles
        if self._settings.image_embeddings_dim > 0:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()
            self._activation_size = None
        
        # Reset state
        self._seen = None
        self._training_phase = None
        self._final_validation = None
