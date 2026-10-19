"""Full-suite coverage, condition routing, leakage and unbiased aggregation."""

import json
from types import SimpleNamespace
import pytest
from tactalent.roboicl.benchmarks import binding, all_bindings
from tactalent.roboicl.evaluation import cases, plan, apply_case, validate_episode, summary
from tactalent.roboicl.suite import job_name, command, validate_catalog


def catalog(tmp_path):
    p = tmp_path / "refs.json"
    p.write_text(
        json.dumps(
            {
                "episodes": [
                    dict(
                        benchmark=s["benchmark"],
                        task=s["task"],
                        variant=s["variant"],
                        icl_mode="vision_tactile",
                        reference="/reference/" + str(i),
                    )
                    for i, s in enumerate(all_bindings())
                ]
            }
        )
    )
    return p


def test_full_plan_covers_19_tasks_and_2100_unique_episodes(tmp_path):
    p = plan(catalog(tmp_path))
    assert p["task_count"] == 19 and p["episode_count"] == 2100
    assert len({job_name(j) for j in p["episodes"]}) == 2100
    for benchmark, expected in (("UniVTAC", 800), ("OpenViTac", 1300)):
        rows = [j for j in p["episodes"] if j["benchmark"] == benchmark]
        assert len(rows) == expected
        for task in {j["task"] for j in rows}:
            assert sum(j["task"] == task for j in rows) == (
                300 if task == "insert_block_v2" else 100
            )
    assert len(p["cases"]["UniVTAC"]) == 8
    assert len(p["cases"]["OpenViTac"]) == 22


def test_current_can_and_block_conditions_are_balanced():
    cs = cases("OpenViTac")
    cans = [c for c in cs if c["task"] == "can_empty_select"]
    assert {c["conditions"]["empty_can"] for c in cans} == {"fanta", "pepsi"}
    assert [c["episodes"] for c in cans] == [50, 50]
    v1 = [c for c in cs if c["task"] == "insert_block_v1"]
    assert len(v1) == 2 and all(c["conditions"]["target_block"] == "cube" for c in v1)
    v2 = [c for c in cs if c["task"] == "insert_block_v2"]
    assert {(c["variant"], tuple(c["conditions"]["block_base_pose_indices"])) for c in v2} == {
        (shape, layout)
        for shape in ("cube", "half_cylinder", "hexagon")
        for layout in ((0, 1, 4), (1, 3, 2))
    }


def test_apply_native_conditions_and_fail_if_unsupported():
    spec = binding("OpenViTac", "insert_block_v1")
    cfg = SimpleNamespace(target_block="half_cylinder", block_base_pose_indices=None, step_lim=300)
    apply_case(cfg, spec, "insert_block_v1_cube_layout_014")
    assert cfg.target_block == "cube" and cfg.block_base_pose_indices == (0, 1, 4)
    assert cfg.step_lim == 300
    with pytest.raises(ValueError, match="does not support"):
        apply_case(SimpleNamespace(), spec, "insert_block_v1_cube_layout_014")
    with pytest.raises(ValueError, match="does not match"):
        apply_case(cfg, spec, "can_empty_select_fanta")


def test_seed_range_and_train_overlap_are_rejected():
    spec = binding("OpenViTac", "can_empty_select")
    case = "can_empty_select_fanta"
    validate_episode(spec, 10000, case, {"demonstrations": [{"seed": 1}, {"seed": 8}]})
    for seed in (9999, 10050, 1):
        with pytest.raises(ValueError, match="outside"):
            validate_episode(spec, seed, case)
    with pytest.raises(ValueError, match="overlaps"):
        validate_episode(spec, 10000, case, {"demonstrations": [{"seed": 10000}]})
    with pytest.raises(ValueError, match="held out"):
        validate_episode(spec, 10000)
    validate_episode(spec, 1000000)  # Existing one-per-task smoke/diagnostic behavior.


def test_case_reaches_single_episode_cli_and_names_do_not_collide(tmp_path):
    jobs = plan(catalog(tmp_path))["episodes"]
    for j in jobs:
        cmd = command(j, tmp_path / job_name(j))
        assert cmd[cmd.index("--evaluation-case") + 1] == j["evaluation_case"]
        assert cmd[cmd.index("--seed") + 1] == str(j["seed"])


def test_catalog_rejects_condition_tampering_and_duplicates(tmp_path, monkeypatch):
    import tactalent.roboicl.suite as suite

    monkeypatch.setattr(suite, "preflight", lambda args: (None, {"demonstrations": []}))
    p = plan(catalog(tmp_path), "OpenViTac")
    assert validate_catalog(p, "live") == {"episodes": 1300, "references": 13}
    p["episodes"][0]["conditions"] = {"weight_label": "unknown"}
    with pytest.raises(ValueError, match="conditions"):
        validate_catalog(p, "live")
    p = plan(catalog(tmp_path), "UniVTAC")
    p["episodes"].append(p["episodes"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_catalog(p, "live")


def test_macro_mean_does_not_triple_weight_ib_v2_or_hide_errors():
    rows = [
        dict(benchmark="OpenViTac", task="insert_USB", result={"status": "success"})
        for _ in range(100)
    ]
    rows += [
        dict(benchmark="OpenViTac", task="insert_block_v2", result={"status": "failed"})
        for _ in range(300)
    ]
    s = summary(rows, "live")
    assert s["macro_success_rate"] == 0.5 and s["valid_success_rate"] == 0.25
    rows += [dict(benchmark="OpenViTac", task="grasp_fragile_chip", result={"status": "error"})]
    assert summary(rows, "live")["macro_success_rate"] is None
    rows.append(
        dict(benchmark="OpenViTac", task="grasp_fragile_chip", result={"status": "success"})
    )
    s = summary(rows, "live")
    chip = next(t for t in s["tasks"] if t["task"] == "grasp_fragile_chip")
    assert chip["valid_success_rate"] == 1 and chip["counts"]["error"] == 1


def test_preflight_routes_condition_validation(monkeypatch):
    from tactalent.roboicl.run import preflight

    args = SimpleNamespace(
        benchmark="OpenViTac",
        task="can_empty_select",
        variant=None,
        reference=None,
        backend="smoke",
        seed=10000,
        icl_mode="vision_tactile",
        evaluation_case="can_empty_select_fanta",
    )
    assert preflight(args)[0]["task"] == "can_empty_select"
    args.evaluation_case = "hardness_classify_soft"
    with pytest.raises(ValueError, match="does not match"):
        preflight(args)
