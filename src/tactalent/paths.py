"""Installed package resources and user-owned experiment workspace."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "_vendor" / "RoboICL"


def workspace_root():
    return Path(os.environ.get("ASTRA_STORAGE_ROOT", str(Path.cwd()))).expanduser().resolve()
