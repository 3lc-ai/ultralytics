# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc
import torch
from typing import Literal

from ultralytics.utils.tlc.settings import Settings


def create_sampler(
    table: tlc.Table, mode: Literal["train", "val"], settings: Settings, distributed: bool = False
) -> torch.utils.data.Sampler | None:
    """Get the sampler for the dataset.

    :param table: The table to get the sampler for.
    :param mode: The mode of the sampler.
    :param settings: The settings for the run.
    :param distributed: Whether training is distributed.
    :returns: The sampler for the dataset.
    """
    sampler = None

    if mode == "train":
        if settings.sampling_weights or settings.exclude_zero_weight_training:
            if distributed:
                raise NotImplementedError("Distributed training and using 3LC weights is not yet supported.")

            try:
                sampler = table.create_sampler(
                    exclude_zero_weights=settings.exclude_zero_weight_training,
                    weighted=settings.sampling_weights,
                    shuffle=True,
                )
            except Exception as e:
                raise ValueError(f"Error creating sampler for table {table.url}") from e

    elif mode == "val":
        if distributed:
            raise NotImplementedError("Distributed validation and exclusion by weight is not yet supported.")

        # Exclude zero weight is handled in the dataset for validation
        return None
    return sampler
