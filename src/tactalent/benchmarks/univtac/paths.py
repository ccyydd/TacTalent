"""Deployment paths for offline UniVTAC training-data preparation."""

import os
from tactalent.paths import workspace_root
from pathlib import Path

PROJECT_ROOT = workspace_root()


def native_root():
    return Path(
        os.environ.get("ASTRA_UNIVTAC_ROOT", "/opt/tactalent/benchmarks/UniVTAC")
    ).expanduser()


def storage_root():
    return Path(os.environ.get("ASTRA_STORAGE_ROOT", str(PROJECT_ROOT))).expanduser()


def source_config():
    storage = storage_root()
    return Path(
        os.environ.get(
            "ASTRA_UNIVTAC_SOURCE_CONFIG", str(storage / "work/training_sources/univtac.yaml")
        )
    ).expanduser()


def urdf_path(root=None):
    root = native_root() if root is None else Path(root)
    return Path(
        os.environ.get("ASTRA_UNIVTAC_URDF", str(root / "assets/embodiments/franka/panda.urdf"))
    ).expanduser()


def asset_package():
    root = Path(os.environ.get("ASTRA_UNIVTAC_TRAIN_CAMERA_ROOT", str(native_root()))).expanduser()
    return Path(
        os.environ.get(
            "ASTRA_UNIVTAC_TACEX_ASSETS",
            str(root / "third_party/TacEx/source/tacex_assets/tacex_assets"),
        )
    ).expanduser()
