"""Pinned RoboICL runtime and benchmark reference fingerprints."""

import hashlib
import json
import sys
from pathlib import Path
from tactalent.paths import PACKAGE_ROOT, VENDOR_ROOT

VENDOR = VENDOR_ROOT
sys.path.insert(0, str(VENDOR))


def verify_vendor():
    hashes = json.loads((VENDOR / "SOURCE_HASHES.json").read_text())
    for relative, expected in hashes.items():
        path = VENDOR / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned RoboICL runtime changed: " + relative)
    return hashes


def identity():
    return json.loads((VENDOR / "UPSTREAM.json").read_text())


def protocol_sha256(spec=None):
    from .benchmarks import binding, native_source_paths
    from .calibration import authored

    spec = spec or binding()
    names = [
        "roboicl/" + name
        for name in (
            "policy.py",
            "provider_errors.py",
            "reference.py",
            "calibration.py",
            "controller.py",
            "prepare_recorded.py",
            "benchmarks.py",
            "environment.py",
            "native_settings.py",
            "evaluation.py",
            "run.py",
            "checkpoint.py",
            "upstream.py",
            "observations.py",
            "demonstrations.py",
            "prepare_few_shot.py",
        )
    ]
    names += [
        "robot/state.py",
        "robot/ik.py",
        "robot/transforms.py",
        "data/kinematics.py",
        "utils/storage.py",
        "utils/api.py",
        "benchmarks/cameras.py",
        "paths.py",
    ]
    owner = "benchmarks/" + spec["benchmark"].lower()
    names.append(owner + "/profile.py")
    if spec["benchmark"] == "UniVTAC":
        names += [owner + "/paths.py", owner + "/sources.py", owner + "/provenance.py"]
    else:
        names += [owner + "/recordings.py", owner + "/scene_seed.py"]
        if spec["task"] == "grasp_fragile_chip":
            names += ["roboicl/openvitac_chip.py", "roboicl/chip_evaluation.py"]
    files = {name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest() for name in names}
    for path in native_source_paths(spec):
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(
        json.dumps(
            {
                "official": identity(),
                "official_runtime": verify_vendor(),
                "binding": files,
                "task": spec,
                "training_calibration": authored(spec, training=True),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def collection_sources(spec):
    """Verify the framework and record exact native training-source hashes."""
    verify_vendor()
    from .benchmarks import native_source_paths

    paths = native_source_paths(spec)
    paths.append(Path(spec["root"]) / "scripts/collect_data.py")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
