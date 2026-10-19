"""Deterministic stratified subsets of reviewed full evaluation plans.

This selects episodes only; it does not change model inputs or native execution.
"""

import copy
import hashlib
import json
from fractions import Fraction
from math import gcd
from functools import reduce

METHOD = "stratified_hash_rank_without_replacement_v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_ratio(value):
    try:
        text = str(value).strip()
        ratio = Fraction(text[:-1]) / 100 if text.endswith("%") else Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ValueError("Sample ratio must be a fraction in (0, 1], e.g. 0.1 or 10%") from None
    if not 0 < ratio <= 1:
        raise ValueError("Sample ratio must be in (0, 1]")
    return ratio


def _key(job):
    return job["benchmark"], job["evaluation_case"]


def _selected_indices(benchmark, case, seed, count):
    # Stable across Python versions, benchmark filtering and case iteration order.
    ranked = sorted(
        range(case["episodes"]),
        key=lambda index: digest([METHOD, seed, benchmark, case["id"], index]),
    )
    return sorted(ranked[:count])


def sample_plan(catalog, ratio, seed=42):
    ratio = parse_ratio(ratio)
    if type(seed) is not int or seed < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    if catalog.get("sampling"):
        raise ValueError(
            "Cannot resample a subset; sample the full plan or reuse its saved catalog"
        )
    groups = catalog.get("cases")
    if not groups:
        raise ValueError("Sampling requires a reviewed full plan with condition groups")
    definitions = {(bench, case["id"]): case for bench, rows in groups.items() for case in rows}
    if not definitions:
        raise ValueError("Cannot sample an empty plan")
    unit = reduce(gcd, (case["episodes"] for case in definitions.values()))
    if any((ratio * case["episodes"]).denominator != 1 for case in definitions.values()):
        raise ValueError(
            f"Ratio {float(ratio):g} gives fractional episodes in a condition group. "
            f"To preserve proportions exactly, use multiples of 1/{unit} "
            f"({100 / unit:g}%), without rounding."
        )
    populations = {}
    for job in catalog["episodes"]:
        populations.setdefault(_key(job), []).append(job)
    if set(populations) != set(definitions):
        raise ValueError("Sampling requires every full-plan condition group")
    selected, strata = set(), []
    for key, case in definitions.items():
        rows = populations[key]
        expected = set(range(case["start_seed"], case["start_seed"] + case["episodes"]))
        if len(rows) != case["episodes"] or {j["seed"] for j in rows} != expected:
            raise ValueError(
                "Sampling requires the complete seed population of each condition group"
            )
        n = int(ratio * case["episodes"])
        seeds = [case["start_seed"] + i for i in _selected_indices(key[0], case, seed, n)]
        selected.update((key[0], key[1], s) for s in seeds)
        strata.append(
            dict(
                benchmark=key[0],
                evaluation_case=key[1],
                population=case["episodes"],
                selected=n,
                selected_seeds=seeds,
            )
        )
    result = copy.deepcopy(catalog)
    result["episodes"] = [
        job for job in result["episodes"] if (*_key(job), job["seed"]) in selected
    ]
    result["episode_count"] = len(result["episodes"])
    result["sampling"] = dict(
        method=METHOD,
        ratio=str(ratio),
        fraction=float(ratio),
        seed=seed,
        population_episode_count=len(catalog["episodes"]),
        selected_episode_count=len(result["episodes"]),
        strata=strata,
        parent_plan_sha256=catalog["plan_sha256"],
        selected_episodes_sha256=digest(result["episodes"]),
    )
    result["plan_sha256"] = digest(result["sampling"])
    return result


def validate_sampling(catalog):
    """Protect saved subsets against silent edits when passed back via --catalog."""
    info = catalog.get("sampling")
    if info is None:
        return
    if info["method"] != METHOD:
        raise ValueError("Unsupported sampling method")
    if (
        info["selected_episodes_sha256"] != digest(catalog["episodes"])
        or catalog["plan_sha256"] != digest(info)
        or catalog["episode_count"] != len(catalog["episodes"])
    ):
        raise ValueError("Sampled catalog differs from its saved selection")
    ratio = parse_ratio(info["ratio"])
    definitions = {(b, c["id"]): c for b, cs in catalog["cases"].items() for c in cs}
    actual = {}
    for job in catalog["episodes"]:
        actual.setdefault(_key(job), []).append(job["seed"])
    if set(actual) != set(definitions):
        raise ValueError("Sampled catalog is missing condition groups")
    expected_strata = []
    for key, case in definitions.items():
        count = ratio * case["episodes"]
        if count.denominator != 1:
            raise ValueError("Sampled catalog has fractional group counts")
        seeds = [
            case["start_seed"] + i
            for i in _selected_indices(key[0], case, info["seed"], int(count))
        ]
        if sorted(actual[key]) != seeds:
            raise ValueError(
                "Sampled catalog does not match its sampling seed or group proportions"
            )
        expected_strata.append(
            dict(
                benchmark=key[0],
                evaluation_case=key[1],
                population=case["episodes"],
                selected=int(count),
                selected_seeds=seeds,
            )
        )
    if (
        info["strata"] != expected_strata
        or info["selected_episode_count"] != len(catalog["episodes"])
        or info["population_episode_count"] != sum(c["episodes"] for c in definitions.values())
        or info["fraction"] != float(ratio)
    ):
        raise ValueError("Sampling audit metadata differs from the selected groups")
