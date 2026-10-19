"""Subset coverage, reproducibility and saved-plan replay without model calls."""

import copy
import json
from collections import Counter
from fractions import Fraction
import pytest
from tactalent.roboicl.benchmarks import all_bindings
from tactalent.roboicl.evaluation import plan
from tactalent.roboicl.sampling import sample_plan, validate_sampling
from tactalent.roboicl.suite import job_name


@pytest.fixture
def reference_catalog(tmp_path):
    path = tmp_path / "references.json"
    path.write_text(
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
    return path


def identities(p):
    return {job_name(j) for j in p["episodes"]}


def test_ten_percent_preserves_all_tasks_conditions_and_layouts(reference_catalog):
    full = plan(reference_catalog)
    before = copy.deepcopy(full)
    p = sample_plan(full, "10%", 42)
    assert full == before
    assert p["episode_count"] == 210 and p["task_count"] == 19
    assert Counter(j["benchmark"] for j in p["episodes"]) == {"UniVTAC": 80, "OpenViTac": 130}
    assert len(identities(p)) == 210 and identities(p) < identities(full)
    for row in p["sampling"]["strata"]:
        assert Fraction(row["selected"], row["population"]) == Fraction(1, 10)
    tasks = Counter((j["benchmark"], j["task"]) for j in p["episodes"])
    assert len(tasks) == 19
    assert all(n == (30 if task == "insert_block_v2" else 10) for (_, task), n in tasks.items())
    validate_sampling(p)


def test_selection_is_reproducible_nested_and_filter_independent(reference_catalog):
    full = plan(reference_catalog)
    a = sample_plan(full, ".1", 42)
    assert a == sample_plan(full, "10%", 42)
    b = sample_plan(full, ".1", 43)
    assert identities(a) != identities(b)
    assert a["plan_sha256"] != b["plan_sha256"]
    assert identities(a) < identities(sample_plan(full, ".2", 42))
    just_open = sample_plan(plan(reference_catalog, "OpenViTac"), ".1", 42)
    assert identities(just_open) == {
        job_name(j) for j in a["episodes"] if j["benchmark"] == "OpenViTac"
    }
    # Changing modality/reference must not change paired evaluation seeds.
    alternate = copy.deepcopy(full)
    for job in alternate["episodes"]:
        job["icl_mode"] = "vision"
        job["reference"] += "_vision"
    selection = lambda p: [(j["benchmark"], j["evaluation_case"], j["seed"]) for j in p["episodes"]]
    assert selection(a) == selection(sample_plan(alternate, ".1", 42))
    first_group = a["sampling"]["strata"][0]
    assert first_group["selected_seeds"] != list(range(1000000, 1000010))


def test_all_and_minimum_ratios_preserve_coverage(reference_catalog):
    full = plan(reference_catalog)
    assert sample_plan(full, "1", 0)["episodes"] == full["episodes"]
    assert sample_plan(full, "0.02")["episode_count"] == 42
    uni = sample_plan(plan(reference_catalog, "UniVTAC"), "5%")
    assert uni["episode_count"] == 40
    validate_sampling(uni)


@pytest.mark.parametrize("ratio", ["0", "-0.1", "1.01", "nan", "inf", "abc", "1/0"])
def test_invalid_ratios_fail_before_execution(reference_catalog, ratio):
    with pytest.raises(ValueError, match="ratio|Ratio"):
        sample_plan(plan(reference_catalog), ratio)


def test_fractional_counts_or_partial_populations_cannot_silently_change_balance(reference_catalog):
    full = plan(reference_catalog)
    for value in (".05", ".01", ".333"):
        with pytest.raises(ValueError, match=r"multiples of 1/50"):
            sample_plan(full, value)
    with pytest.raises(ValueError, match="nonnegative"):
        sample_plan(full, ".1", -1)
    with pytest.raises(ValueError, match="resample"):
        sample_plan(sample_plan(full, ".1"), ".5")
    full["episodes"].pop()
    with pytest.raises(ValueError, match="complete seed population"):
        sample_plan(full, ".1")


def test_saved_catalog_validates_and_rejects_modified_selection(reference_catalog, monkeypatch):
    from tactalent.roboicl.suite import validate_catalog
    import tactalent.roboicl.suite as suite

    monkeypatch.setattr(suite, "preflight", lambda args: (None, {"demonstrations": []}))
    selected = sample_plan(plan(reference_catalog), ".1")
    saved = json.loads(json.dumps(selected))
    assert validate_catalog(saved, "live") == {"episodes": 210, "references": 21}
    saved["episodes"].pop()
    with pytest.raises(ValueError, match="saved selection"):
        validate_catalog(saved, "live")
    saved = copy.deepcopy(selected)
    saved["sampling"]["seed"] += 1
    with pytest.raises(ValueError, match="saved selection"):
        validate_catalog(saved, "live")


def test_cli_plan_saves_selection_without_execution(reference_catalog, tmp_path, monkeypatch):
    import tactalent.roboicl.suite as suite

    monkeypatch.setattr(suite, "preflight", lambda args: (None, {"demonstrations": []}))
    monkeypatch.setattr(suite, "require_output_path", lambda path: path)
    monkeypatch.setattr(
        suite.subprocess, "run", lambda *a, **k: pytest.fail("plan-only launched a process")
    )
    out = tmp_path / "sample"
    monkeypatch.setattr(
        "sys.argv",
        [
            "suite",
            "--reference-catalog",
            str(reference_catalog),
            "--sample-ratio",
            ".1",
            "--sample-seed",
            "7",
            "--plan-only",
            "--output",
            str(out),
        ],
    )
    suite.main()
    saved = json.loads((out / "catalog.snapshot.json").read_text())
    assert saved == sample_plan(plan(reference_catalog), ".1", 7)
    replay = tmp_path / "replay"
    monkeypatch.setattr(
        "sys.argv",
        [
            "suite",
            "--catalog",
            str(out / "catalog.snapshot.json"),
            "--plan-only",
            "--output",
            str(replay),
        ],
    )
    suite.main()
    assert json.loads((replay / "catalog.snapshot.json").read_text()) == saved
