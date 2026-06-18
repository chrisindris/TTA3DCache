"""Training utilities for PyTorch experiments."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
