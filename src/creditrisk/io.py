"""Small, explicit artifact and provenance helpers."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False), encoding="utf-8"
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def new_directory(path):
    """Never silently replace an existing run or dataset."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path
