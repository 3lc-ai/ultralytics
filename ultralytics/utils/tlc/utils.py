# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from .constants import TRAINING_PHASE

from ultralytics.utils import LOGGER
from ultralytics.utils.tlc.constants import TLC_COLORSTR, TLC_REQUIRED_VERSION

from packaging import version

def training_phase_schema() -> tlc.Schema:
    """Create a 3LC schema for the training phase.

    :returns: The training phase schema.
    """
    return tlc.Schema(
        display_name=TRAINING_PHASE,
        description=("'During' metrics are collected with EMA during training, "
                     "'After' is with the final model weights after completed training."),
        display_importance=tlc.DISPLAY_IMPORTANCE_EPOCH - 1,  # Right hand side of epoch in the Dashboard
        writable=False,
        computable=False,
        value=tlc.Int32Value(
            value_min=0,
            value_max=1,
            value_map={
                float(0): tlc.MapElement(display_name='During'),
                float(1): tlc.MapElement(display_name='After'), },
        ))

def image_embeddings_schema(activation_size=512) -> dict[str, tlc.Schema]:
    """ Create a 3LC schema for YOLOv8 image embeddings.

    :param activation_size: The size of the activation tensor.
    :returns: The YOLO image embeddings schema.
    """
    embedding_schema = tlc.Schema('Embedding',
                                  'Large NN embedding',
                                  writable=False,
                                  computable=False,
                                  value=tlc.Float32Value(number_role=tlc.NUMBER_ROLE_NN_EMBEDDING),
                                  size0=tlc.DimensionNumericValue(value_min=activation_size,
                                                                  value_max=activation_size,
                                                                  enforce_min=True,
                                                                  enforce_max=True))
    return embedding_schema

def reduce_embeddings(
    run: tlc.Run,
    method: str,
    n_components: int,
    foreign_table_url: tlc.Url | None = None,
):
    """Reduce image embeddings by a foreign table URL."""
    if foreign_table_url is None:
        foreign_table_url = tlc.Url(
            tlc.active_run().constants['inputs'][0]['input_table_url']
        ).to_absolute(tlc.active_run().url)

    LOGGER.info(TLC_COLORSTR + f"Reducing image embeddings to {n_components}D with {method}, this may take a few minutes...")
    run.reduce_embeddings_by_foreign_table_url(
        foreign_table_url=foreign_table_url,
        method=method,
        n_components=n_components,
    )

def check_tlc_version():
    """Check the 3LC version."""
    installed_version = version.parse(tlc.__version__)
    if installed_version < version.parse(TLC_REQUIRED_VERSION):
        raise ValueError(
            f"3LC version {tlc.__version__} is too old to use the YOLOv8 integration. "
            f"Please upgrade to version {TLC_REQUIRED_VERSION} or later by running 'pip install --upgrade 3lc'."
    )