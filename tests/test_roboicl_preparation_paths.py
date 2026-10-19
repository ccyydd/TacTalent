"""Portable source preparation with real HDF5 conversion, without Isaac Sim or API calls."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tactalent.roboicl.prepare_bank import (
    SCHEMA,
    build,
    checked_catalog,
    source_jobs,
    source_template,
    training_config,
)
from tactalent.roboicl.evaluation import default_reference_catalog
from tactalent.benchmarks.univtac import paths as source_paths, sources as training_source
from tactalent.utils import storage


@pytest.fixture
def training(tmp_path, monkeypatch):
    native = tmp_path / "UniVTAC"
    for directory in [
        "envs/robot",
        "policy",
        "task_config",
        "scripts",
        "assets/embodiments/franka",
    ]:
        (native / directory).mkdir(parents=True, exist_ok=True)
    (native / "policy/task_settings.json").write_text(
        json.dumps({"pull_out_key": {"camera_type": "head"}})
    )
    (native / "envs/pull_out_key.py").write_text("class TaskCfg:\n    pass\n")
    (native / "envs/_base_task.py").write_text(
        "class BaseTaskCfg:\n    cameras = [CameraCfg(name='head', width=480, height=270, "
        "offset=OffsetCfg(pos=(0,0,0), rot=(1,0,0,0), convention='opengl'), "
        "spawn=CameraSpawn(focal_length=24, horizontal_aperture=36))]\n"
    )
    (native / "envs/robot/robot.py").write_text("# synthetic robot\n")
    (native / "task_config/demo.yml").write_text("force: true\n")
    (native / "scripts/collect_data.py").write_text(
        "def collect(task):\n    if task.plan_success and task.check_success() and not task.check_early_stop():\n"
        "        task.save_to_hdf5()\n"
    )
    joints = [
        f'<joint name="panda_joint{i}" type="revolute"><parent link="panda_link{i-1}"/>'
        f'<child link="panda_link{i}"/><axis xyz="0 0 1"/></joint>'
        for i in range(1, 8)
    ]
    joints.append(
        '<joint name="hand" type="fixed"><parent link="panda_link7"/><child link="panda_hand"/></joint>'
    )
    (native / "assets/embodiments/franka/panda.urdf").write_text(
        '<robot name="synthetic">' + "".join(joints) + "</robot>"
    )
    data = tmp_path / "data"
    episode = data / "pull_out_key/clean/0.hdf5"
    episode.parent.mkdir(parents=True)
    count = 66
    with h5py.File(episode, "w") as file:
        file["step"] = np.arange(count, dtype=np.int64) * 2
        ee = np.zeros((count, 7), dtype=np.float32)
        ee[:, 3] = 1
        q = np.zeros((count, 9), dtype=np.float32)
        q[:, 7:] = 0.0195
        file["embodiment/ee"] = ee
        file["embodiment/joint"] = q
        for name, shape in [
            ("observation/head/rgb", (270, 480, 3)),
            ("tactile/left_tactile/rgb", (240, 320, 3)),
            ("tactile/right_tactile/rgb", (240, 320, 3)),
        ]:
            file.create_dataset(name, shape=(count, *shape), dtype=np.uint8, fillvalue=128)
    digest = hashlib.sha256(episode.read_bytes()).hexdigest()
    (data / "download_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "pull_out_key/clean/0.hdf5": {
                        "verified_sha256": digest,
                        "upstream_sha256": digest,
                        "size": episode.stat().st_size,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(storage, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("ASTRA_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("ASTRA_UNIVTAC_ROOT", str(native))
    monkeypatch.delenv("ASTRA_UNIVTAC_URDF", raising=False)
    monkeypatch.delenv("ASTRA_REFERENCE_CATALOG", raising=False)
    from tactalent.benchmarks.univtac import profile as roboicl_univtac
    from tactalent.benchmarks.openvitac import profile as roboicl_openvitac

    monkeypatch.setattr(roboicl_univtac, "ROOT", native)
    monkeypatch.setattr(roboicl_univtac, "TRAIN_CAMERA_ROOT", native)
    # A UniVTAC-only setup must not accidentally access the other benchmark.
    monkeypatch.setattr(roboicl_openvitac, "ROOT", tmp_path / "OpenViTac-not-installed")
    config = training_config(data, tmp_path / "source.yaml")
    monkeypatch.setenv("ASTRA_UNIVTAC_SOURCE_CONFIG", str(config))
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "tasks": [{"benchmark": "UniVTAC", "task": "pull_out_key", "training_episode": 0}],
            }
        )
    )
    return dict(native=native, data=data, episode=episode, sources=sources, config=config)


def test_configured_paths_and_environment_overrides(training, tmp_path, monkeypatch):
    assert source_paths.native_root() == training["native"]
    assert source_paths.source_config() == training["config"]
    assert source_paths.urdf_path() == training["native"] / "assets/embodiments/franka/panda.urdf"
    monkeypatch.setenv("ASTRA_UNIVTAC_URDF", str(tmp_path / "custom.urdf"))
    monkeypatch.setenv("ASTRA_UNIVTAC_TACEX_ASSETS", str(tmp_path / "custom-assets"))
    assert source_paths.urdf_path() == tmp_path / "custom.urdf"
    assert source_paths.asset_package() == tmp_path / "custom-assets"


def test_training_import_uses_configured_native_and_data_roots(training):
    result = training_source.build("pull_out_key")
    assert result["source"]["episode_path"] == str(training["episode"])
    assert result["robot_audit"]["fk_checked_frames"] == 66
    assert result["source"]["success_evidence"]["collector_path"].startswith(
        str(training["native"])
    )


def test_training_hash_and_split_checks_remain_enforced(training):
    with pytest.raises(ValueError, match="training split"):
        training_source.build("pull_out_key", episode_id=50)
    manifest = training["data"] / "download_manifest.json"
    data = json.loads(manifest.read_text())
    data["files"]["pull_out_key/clean/0.hdf5"]["upstream_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="download manifest"):
        training_source.build("pull_out_key")


@pytest.mark.parametrize("mode", ["vision", "vision_tactile"])
def test_real_conversion_and_catalog_preflight_need_no_second_benchmark(
    training, tmp_path, monkeypatch, mode
):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: pytest.fail("model/network call during preparation"),
    )
    output = tmp_path / mode
    result = build(training["sources"], mode, output)
    assert result["api_calls"] == 0 and result["references"] == 1
    catalog = json.loads(Path(result["catalog"]).read_text())
    ref = json.loads(Path(catalog["episodes"][0]["reference"]).read_text())
    assert ref["horizon"] == 5 and len(ref["starts"]) == 12
    assert ref["frequency"] == 60 and ref["action_repeat"] == 2
    assert ("tactile_left" in json.dumps(ref["messages"])) == (mode == "vision_tactile")
    assert catalog["episodes"][0]["icl_mode"] == mode
    other = "vision" if mode == "vision_tactile" else "vision_tactile"
    with pytest.raises(ValueError):
        checked_catalog(output, other, tmp_path / "wrong-mode.json")
    assert not (tmp_path / "wrong-mode.json").exists()
    ref["protocol_sha256"] = "stale"
    Path(catalog["episodes"][0]["reference"]).write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="version mismatch"):
        checked_catalog(output, mode, tmp_path / "stale.json")
    assert not (tmp_path / "stale.json").exists()


def test_check_only_and_duplicate_source_validation(training, tmp_path):
    output = tmp_path / "check-only"
    assert build(training["sources"], "vision", output, check_only=True)["tasks_and_variants"] == 1
    assert not output.exists()
    source = json.loads(training["sources"].read_text())
    source["tasks"] *= 2
    training["sources"].write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Repeated source"):
        build(training["sources"], "vision", output)
    assert not output.exists()


def test_templates_cover_19_logical_tasks_and_preserve_class_ledger_requirements(
    training, tmp_path
):
    path = source_template(tmp_path / "all.json")
    rows = json.loads(path.read_text())["tasks"]
    assert len(rows) == 19
    assert sum(row["task"] == "insert_block_v2" for row in rows) == 1
    assert all(
        "training_ledger" in row
        for row in rows
        if row["benchmark"] == "OpenViTac" or row["task"] == "grasp_classify"
    )
    assert all(row["training_episode"] == 0 for row in rows if "training_episode" in row)


def test_mode_catalog_defaults_and_explicit_override(training, tmp_path, monkeypatch):
    assert default_reference_catalog("vision") == tmp_path / "references/vision/catalog.json"
    assert (
        default_reference_catalog("vision_tactile")
        == tmp_path / "references/vision_tactile/catalog.json"
    )
    monkeypatch.setenv("ASTRA_REFERENCE_CATALOG", str(tmp_path / "custom/refs.json"))
    assert default_reference_catalog("vision") == tmp_path / "custom/refs.json"


def test_prepare_does_not_overwrite_a_bank_or_source_declaration(training, tmp_path):
    with pytest.raises(FileExistsError):
        training_config(training["data"], training["config"])
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build(training["sources"], "vision", output)


def test_missing_and_evaluation_training_identity_fail_before_creating_outputs(training, tmp_path):
    data = json.loads(training["sources"].read_text())
    data["tasks"][0]["training_episode"] = 1000000
    training["sources"].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="0..49"):
        build(training["sources"], "vision", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_cli_build_honors_custom_catalog_filename(training, tmp_path, monkeypatch, capsys):
    from tactalent.roboicl.prepare_bank import main

    catalog = tmp_path / "custom_bank/refs.json"
    monkeypatch.setenv("ASTRA_REFERENCE_CATALOG", str(catalog))
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_bank", "build", "--sources", str(training["sources"]), "--icl-mode", "vision"],
    )
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["catalog"] == str(catalog) and catalog.is_file()
    assert not catalog.with_name("catalog.json").exists()
