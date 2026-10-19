"""Verified OpenViTac training ledgers and native RGB decoding."""

import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .profile import UPSTREAM_COMMIT as COMMIT
from tactalent.utils.storage import require_output_path
from tactalent.data.kinematics import validate_episode_arrays as validate_robot_arrays


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ledger(path, config):
    path = require_output_path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    if (
        data.get("benchmark") != "OpenViTac"
        or data.get("task") != config["task"]
        or data.get("variant") != config.get("variant")
        or data.get("split") != "train"
        or data.get("upstream_commit") != COMMIT
        or not data.get("split_basis")
        or not isinstance(data.get("instruction"), str)
    ):
        raise ValueError("Explicit task-matched training provenance required")
    rows = data.get("episodes", [])
    if not rows:
        raise ValueError("No training episodes in ledger")
    seeds = set()
    paths = set()
    for row in rows:
        if (
            type(row.get("seed")) is not int
            or not 0 <= row["seed"] < 1000000
            or row["seed"] in seeds
        ):
            raise ValueError(
                "Training seeds must be unique and disjoint from evaluation seeds >=1000000"
            )
        p = require_output_path(row["path"]).resolve(strict=True)
        if p in paths or row.get("success") is not True or sha(p) != row.get("sha256"):
            raise ValueError("Unverified or duplicate training file")
        seeds.add(row["seed"])
        paths.add(p)
    return data


def validate_recorded_arrays(step, ee, joint):
    """Native experts can omit settling frames; preserve every recorded timestamp."""
    step = np.asarray(step)
    if step.ndim != 1 or step.dtype.kind not in "iu" or np.any(step < 0):
        raise ValueError("Expected nonnegative integer native step timestamps")
    if np.any(np.diff(step) <= 0):
        raise ValueError("Expected strictly increasing native step timestamps")
    # Reuse only the audited robot schema/FK prerequisites. This temporary index
    # is not exported or presented to the model; the original native clock is kept.
    _, ee, joint = validate_robot_arrays(np.arange(len(step), dtype=np.int64) * 2, ee, joint)
    return step, ee, joint


def decode(dataset):
    frames = []
    for raw in dataset:
        if np.asarray(raw).ndim == 3:
            image = np.asarray(raw)
        else:
            image = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
        # Upstream img_to_stream passes numeric sensor RGB straight to imencode.
        # Thus imdecode recovers numeric RGB; a BGR-to-RGB swap here would reverse it.
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Invalid original RGB frame")
        frames.append(image)
    return np.stack(frames)
