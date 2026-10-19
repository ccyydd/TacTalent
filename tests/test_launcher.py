"""Launch isolation without starting a simulator or using an API endpoint."""

from pathlib import Path
import sys

import pytest

from tactalent import runtime
from tactalent.paths import PACKAGE_ROOT


@pytest.mark.parametrize("benchmark", ["UniVTAC", "OpenViTac"])
def test_native_paths_follow_selected_benchmark(tmp_path, benchmark):
    key = "ASTRA_UNIVTAC_ROOT" if benchmark == "UniVTAC" else "ASTRA_OPENVITAC_ROOT"
    native = tmp_path / benchmark
    native.mkdir()
    root, env = runtime.child_environment(benchmark, 2, environment={key: str(native)})
    assert root == native
    paths = env["PYTHONPATH"].split(":")
    assert paths[:2] == [str(PACKAGE_ROOT.parent), str(native)]
    assert str(native / "third_party/TacEx/source/tacex") in paths
    assert Path(env["ASTRA_STORAGE_ROOT"]).is_absolute()


def test_dry_run_does_not_acquire_gpu_or_create_runtime(tmp_path, monkeypatch):
    native = tmp_path / "UniVTAC"
    native.mkdir()
    workspace = tmp_path / "outputs"
    monkeypatch.setenv("ASTRA_UNIVTAC_ROOT", str(native))
    monkeypatch.setenv("ASTRA_STORAGE_ROOT", str(workspace))
    monkeypatch.setenv("GPU_ID", "3")
    monkeypatch.setattr(runtime.fcntl, "flock", lambda *a: pytest.fail("dry run locked a GPU"))
    monkeypatch.setattr(
        runtime.subprocess, "check_output", lambda *a, **k: pytest.fail("dry run queried GPU/ports")
    )
    calls = []
    monkeypatch.setattr(runtime.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    with pytest.raises(SystemExit) as result:
        runtime.launch("eval", ["--benchmark", "UniVTAC", "--task", "pull_out_key", "--dry-run"])
    assert result.value.code == 0
    command, options = calls[0]
    assert command[0] == sys.executable
    assert "Image.preinit()" in command[3]
    assert "tactalent.roboicl.run" in command[3]
    assert "--dry-run" in command
    assert not workspace.exists()


def test_unsupported_collection_and_missing_native_root_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="released training"):
        runtime.launch("collect", ["--benchmark", "UniVTAC"])
    with pytest.raises(ValueError, match="ASTRA_OPENVITAC_ROOT"):
        runtime.child_environment("OpenViTac", 0, environment={})
