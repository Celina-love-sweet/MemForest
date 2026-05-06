import json
import os
from pathlib import Path


def load_config(config_path: str | Path = "config.local.json"):
    """
    Load key-value pairs from a JSON file and set them as environment variables
    if they are not already set.
    """
    path = Path(config_path)
    if not path.exists():
        # fall back to example file if present
        example = Path("config.example.json")
        if example.exists():
            path = example
        else:
            return

    try:
        data = json.loads(path.read_text())
    except Exception:
        return

    for k, v in data.items():
        if k and v is not None and k not in os.environ:
            os.environ[k] = str(v)

