"""Public packaging and two-benchmark scope checks."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tactalent.paths import PACKAGE_ROOT
from tactalent.roboicl import upstream


def test_only_two_benchmarks_are_exposed_without_native_installations():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASTRA_")}
    env["PYTHONPATH"] = str(PACKAGE_ROOT.parent)
    result = subprocess.run(
        [sys.executable, "-m", "tactalent", "list"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "UniVTAC (8 tasks)" in result.stdout
    assert "OpenViTac (11 tasks)" in result.stdout
    tasks = [line.strip() for line in result.stdout.splitlines() if line.startswith("  ")]
    assert len(tasks) == len(set(tasks)) == 19
    assert "ManiFeel" not in result.stdout and "SoftVTBench" not in result.stdout


@pytest.mark.parametrize(
    "command", ["eval", "suite", "prepare", "collect", "review", "review-batch", "checkpoint"]
)
def test_public_command_help_does_not_start_a_simulator(command):
    result = subprocess.run(
        [sys.executable, "-m", "tactalent", command, "--help"],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT.parent)),
    )
    assert "usage:" in result.stdout
    assert "SimulationApp" not in result.stdout


def test_pinned_runtime_files_are_verified():
    hashes = upstream.verify_vendor()
    assert "LICENSE" in hashes
    assert "roboicl/policy/dialogue_policy.py" in hashes
    assert not (upstream.VENDOR / "configs/tasks").exists()


def test_changed_vendored_source_is_rejected(tmp_path, monkeypatch):
    original = b"pinned source\n"
    (tmp_path / "source.py").write_bytes(original)
    (tmp_path / "SOURCE_HASHES.json").write_text(
        json.dumps({"source.py": hashlib.sha256(original).hexdigest()})
    )
    monkeypatch.setattr(upstream, "VENDOR", tmp_path)
    upstream.verify_vendor()
    (tmp_path / "source.py").write_text("changed source\n")
    with pytest.raises(ValueError, match="Pinned RoboICL runtime changed"):
        upstream.verify_vendor()


def test_owned_package_has_no_legacy_import_dependencies():
    import ast

    forbidden = (
        "astra_univtac",
        "astra_openvitac",
        "astra_manifeel",
        "astra_softvtbench",
        "astra_demonstrations",
        "astra_benchmarks",
        "benchmark_profiles",
        "tools",
    )
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "_vendor" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in names
                for prefix in forbidden
            ), path
