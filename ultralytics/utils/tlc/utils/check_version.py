# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from packaging import version

from ultralytics.utils.tlc.constants import TLC_REQUIRED_VERSION

def check_tlc_version():
    """Check the 3LC version."""
    installed_version = version.parse(tlc.__version__)
    if installed_version < version.parse(TLC_REQUIRED_VERSION):
        raise ValueError(
            f"3LC version {tlc.__version__} is too old to use the YOLOv8 integration. "
            f"Please upgrade to version {TLC_REQUIRED_VERSION} or later by running 'pip install --upgrade 3lc'.")