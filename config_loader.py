"""
Shared configuration loader.

Loads JSON configuration files from the config/ directory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "config"


@lru_cache(maxsize=None)
def load_config(filename: str) -> dict:
    """
    Load a JSON configuration file.

    Example:
        schema = load_config("schema.json")
    """
    path = CONFIG_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)