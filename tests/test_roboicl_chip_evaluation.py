"""Execute the pinned native lifecycle with real sample transitions, without Isaac."""

import ast
import dataclasses
import math
from types import SimpleNamespace as NS
import pytest
from tactalent.benchmarks.openvitac.profile import ROOT
from tactalent.roboicl.chip_evaluation import PROTOCOL, recovery_scorer, evaluation_task_class


@pytest.fixture
def native():
    path = ROOT / "envs/grasp_fragile_chip.py"
    names = {"ChipAcceptanceLimits", "ChipLifecycle", "ChipPhysicalSample"}
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    ns = dict(dataclass=dataclasses.dataclass, asdict=dataclasses.asdict, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return NS(**ns)


def monitor(native):
    obs = NS(task=NS(fractured=False), reference=object(), grip_qpos=0.02, support_ee_z=0.1)
    obs.scorer = recovery_scorer(native.ChipLifecycle(), obs)
    return obs


def tick(native, scorer, count=1, **values):
    for _ in range(count):
        scorer.advance(native.ChipPhysicalSample(step=scorer.last_step + 1, **values))


SUPPORT = dict(in_target=True, support_contact=True, support_gap_m=0.0, lift_m=0.01)


def grasp_lift(native, scorer):
    tick(native, scorer, 6, grip_contact=True)
    tick(native, scorer, 10, grip_contact=True, lift_m=0.01)
    assert scorer.grasp_verified and scorer.lift_verified


def finish_supported(native, scorer, **changes):
    values = dict(SUPPORT, released=True, withdrawn=True)
    values.update(changes)
    tick(native, scorer, 60, **values)


def test_native_ordered_success_is_preserved(native):
    old, new = native.ChipLifecycle(), monitor(native).scorer
    for scorer in (old, new):
        grasp_lift(native, scorer)
        tick(native, scorer, 14, grip_contact=True, **SUPPORT)
        tick(native, scorer, opening=True, **SUPPORT)
        finish_supported(native, scorer)
        assert scorer.success
    assert old.grasp_verified and new.grasp_verified
    assert old.lift_verified and new.lift_verified
    assert new.snapshot()["release_required"] is False
    assert new.snapshot()["withdrawal_required"] is False


@pytest.mark.parametrize(
    "values,reason",
    [
        (dict(grip_contact=False), "slip"),
        (dict(grip_contact=True, drift_m=0.01), "unstable_grasp"),
        (dict(grip_contact=True, rotation_rad=0.3), "unstable_grasp"),
        (dict(opening=True), "early_release"),
    ],
)
def test_recoverable_events_clear_qualification_and_allow_new_attempt(native, values, reason):
    old, obs = native.ChipLifecycle(), monitor(native)
    for scorer in (old, obs.scorer):
        tick(native, scorer, 6, grip_contact=True)
        tick(native, scorer, **values)
    assert old.failure == reason
    scorer = obs.scorer
    assert not scorer.failure and not scorer.success
    assert scorer.nonterminal_events[-1]["reason"] == reason
    assert not scorer.grasp_verified and not scorer.lift_verified
    assert obs.reference is obs.grip_qpos is obs.support_ee_z is None
    # A loose chip merely ending up on the plate is insufficient.
    finish_supported(native, scorer)
    assert not scorer.success
    grasp_lift(native, scorer)
    tick(native, scorer, opening=True, in_target=True, lift_m=0.01)
    finish_supported(native, scorer)
    assert scorer.success


def test_orientation_before_contact_is_an_event_not_a_terminated_episode(native):
    scorer = monitor(native).scorer
    tick(native, scorer, pose_error_rad=0.1)
    assert not scorer.failure and scorer.stage == "approach"
    assert scorer.nonterminal_events[0]["reason"] == "gripper_orientation_violation"


@pytest.mark.parametrize(
    "changes",
    [
        dict(support_contact=False),
        dict(support_gap_m=0.01),
        dict(speed_m_s=0.02),
        dict(angular_speed_rad_s=0.3),
    ],
)
def test_placement_requires_actual_stable_support(native, changes):
    scorer = monitor(native).scorer
    grasp_lift(native, scorer)
    tick(native, scorer, opening=True, in_target=True, lift_m=0.01)
    assert scorer.stage == "release" and not scorer.release_supported
    finish_supported(native, scorer, **changes)
    assert not scorer.success
    finish_supported(native, scorer)
    assert scorer.success
    assert not scorer.release_supported  # Never invent support before opening.


def test_early_landing_is_explicit_relaxation_of_native_success(native):
    old, new = native.ChipLifecycle(), monitor(native).scorer
    for scorer in (old, new):
        grasp_lift(native, scorer)
        tick(native, scorer, opening=True, in_target=True, lift_m=0.01)
        finish_supported(native, scorer)
    assert old.failure == "early_release" and not old.success
    assert new.success
    assert new.nonterminal_events[-1]["recovery"] == "await_stable_placement"


def test_off_plate_requires_a_new_grasp_lift_attempt(native):
    scorer = monitor(native).scorer
    grasp_lift(native, scorer)
    tick(native, scorer, opening=True, in_target=True, lift_m=0.01)
    tick(native, scorer, released=True, in_target=False)
    assert scorer.nonterminal_events[-1]["reason"] == "off_plate"
    finish_supported(native, scorer)
    assert not scorer.success and not scorer.failure


def test_plate_overpress_gate_does_not_skip_valid_support_samples(native):
    old, new = native.ChipLifecycle(), monitor(native).scorer
    for scorer in (old, new):
        grasp_lift(native, scorer)
        tick(native, scorer, 14, grip_contact=True, overtravel_m=0.001, **SUPPORT)
    assert old.failure == "plate_overpress"
    assert not new.failure and new.support_ticks == 14
    assert new.limits.overtravel_m == 0.0006
    assert new.last_sample.overtravel_m == 0.001
    assert new.snapshot()["plate_overpress_enforced"] is False


@pytest.mark.parametrize("reason", ["", "squeeze", "impact"])
def test_real_fracture_remains_terminal(native, reason):
    scorer = monitor(native).scorer
    tick(native, scorer, pose_error_rad=0.1)
    tick(native, scorer, fractured=True, damage_reason=reason)
    assert scorer.failure == (reason or "fracture")
    finish_supported(native, scorer)
    assert not scorer.success


@pytest.mark.parametrize("values", [dict(valid=False), dict(drift_m=float("nan"))])
def test_invalid_physics_is_an_error(native, values):
    scorer = monitor(native).scorer
    with pytest.raises(RuntimeError, match="physics error"):
        tick(native, scorer, **values)
    assert scorer.outcome == "invalid"


@pytest.mark.parametrize("reason", ["policy_timeout", "unknown_native_reason"])
def test_budgets_and_unreviewed_failure_reasons_stay_terminal(native, reason):
    scorer = monitor(native).scorer
    scorer.fail(reason)
    assert scorer.failure == reason and not scorer.nonterminal_events


def test_finish_does_not_restart_recovery(native):
    scorer = monitor(native).scorer
    tick(native, scorer, pose_error_rad=0.1)
    scorer.finish("incomplete")
    assert scorer.failure == "incomplete"
    assert len(scorer.nonterminal_events) == 1


def test_install_after_evidence_is_forbidden(native):
    scorer = native.ChipLifecycle()
    tick(native, scorer)
    with pytest.raises(RuntimeError, match="before the first policy step"):
        recovery_scorer(scorer, NS())


def test_native_env_step_uses_modified_scorer_without_changing_action_accounting(native):
    import torch
    from typing import Literal

    path = ROOT / "envs/_base_task.py"
    base = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "BaseTask"
    )
    fn = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "env_step")
    ns = dict(torch=torch, Literal=Literal)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)

    class Monitor:
        def __init__(self, task):
            self.task = task
            self.scorer = native.ChipLifecycle()
            self.reference = self.grip_qpos = self.support_ee_z = None

    class Task:
        env_step = ns["env_step"]
        PHASE_POLICY, PHASE_TERMINAL = 1, 2
        device = "cpu"

        def __init__(self):
            self.reset()

        def reset(self):
            self.metadata = {}
            self.fractured = False
            self.phase_id = self.PHASE_POLICY
            self.take_action_cnt = self.policy_step_count = self.physics_steps = 0
            self.cfg = NS(
                step_lim=3600,
                final_policy_timeout_seconds=60,
                policy_action_repeat=2,
                final_acceptance_contract=True,
            )
            self._action_monitor = Monitor(self)

        def take_action(self, *args, action_repeat=None, **kwargs):
            self.take_action_cnt += 1
            for _ in range(action_repeat):
                self._step()
            return True, False

        def _step(self):
            self.physics_steps += 1
            tick(native, self._action_monitor.scorer, pose_error_rad=0.1)
            self.metadata["success_contract"] = "chip_physical_lifecycle_v3"

        def clean_cache(self, mean_steps=0.0, result=None):
            self.saved_outcome = self.metadata["physical_result"]
            self.saved_reason = self.metadata["terminal_reason"]

        def check_early_stop(self):
            return bool(self.fractured or self._action_monitor.scorer.failure)

        def check_rl_early_stop(self, metrics):
            return False

        def get_rl_metrics(self):
            return {}

        def compute_rl_reward(self, *args):
            return 0.0

        def _get_observations(self):
            return {}

        def _set_phase(self, phase, **kwargs):
            self.phase_id = phase

    task = evaluation_task_class(Task, Monitor)()
    _, _, success, truncated, info = task.env_step([0.0] * 8)
    assert not success and not truncated and not task.check_early_stop()
    assert task.take_action_cnt == task.policy_step_count == 1
    assert task.physics_steps == 2
    assert task.cfg.step_lim == 3600 and task.cfg.final_policy_timeout_seconds == 60
    assert task.metadata["success_contract"] == PROTOCOL
    assert task.metadata["evaluation_rules"]["matches_upstream_protocol"] is False
    task.clean_cache(result="failed")
    assert task.saved_outcome == "failure" and task.saved_reason == "incomplete"
    task.reset()
    assert task._action_monitor.scorer.last_step == -1
    task.clean_cache(result="error")
    assert task.saved_outcome == "invalid" and task.saved_reason == "error"
    # Native training class is not mutated.
    original = Task()
    _, _, _, truncated, _ = original.env_step([0.0] * 8)
    assert truncated and original.check_early_stop()


@pytest.mark.parametrize(
    "opening,released,withdrawn",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_placement_success_does_not_require_release_or_withdrawal(
    native, opening, released, withdrawn
):
    scorer = monitor(native).scorer
    grasp_lift(native, scorer)
    values = dict(
        SUPPORT, grip_contact=not released, opening=opening, released=released, withdrawn=withdrawn
    )
    tick(native, scorer, 13, **values)
    assert not scorer.success
    tick(native, scorer, **values)
    assert scorer.success and scorer.stable_placement_ticks == 14
    # No synthetic release/withdrawal values are stored in evidence.
    assert scorer.last_sample.released == released
    assert scorer.last_sample.withdrawn == withdrawn
    tick(native, scorer, 3, **values)
    assert scorer.success  # Native gripper-only post-success gate no longer applies.


def test_stability_dwell_resets_and_duplicate_samples_do_not_count(native):
    scorer = monitor(native).scorer
    grasp_lift(native, scorer)
    tick(native, scorer, 13, grip_contact=True, **SUPPORT)
    scorer.advance(scorer.last_sample)
    assert scorer.stable_placement_ticks == 13 and not scorer.success
    tick(native, scorer, grip_contact=True, speed_m_s=0.02, **SUPPORT)
    assert scorer.stable_placement_ticks == 0
    tick(native, scorer, 13, grip_contact=True, **SUPPORT)
    assert not scorer.success
    tick(native, scorer, grip_contact=True, **SUPPORT)
    assert scorer.success


@pytest.mark.parametrize(
    "changes",
    [
        dict(in_target=False),
        dict(support_contact=False),
        dict(fractured=True, damage_reason="squeeze"),
    ],
)
def test_real_support_loss_or_damage_after_placement_revokes_success(native, changes):
    scorer = monitor(native).scorer
    grasp_lift(native, scorer)
    tick(native, scorer, 14, grip_contact=True, **SUPPORT)
    assert scorer.success
    values = dict(SUPPORT, grip_contact=True)
    values.update(changes)
    tick(native, scorer, **values)
    assert not scorer.success


@pytest.mark.parametrize("angle", [0.1, 0.3])
def test_continuous_orientation_event_does_not_block_grasp_lift_or_placement(native, angle):
    obs = monitor(native)
    scorer = obs.scorer
    reference = obs.reference
    tick(native, scorer, 6, grip_contact=True, pose_error_rad=angle)
    tick(native, scorer, 10, grip_contact=True, lift_m=0.01, pose_error_rad=angle)
    assert scorer.grasp_verified and scorer.lift_verified
    assert obs.reference is reference and obs.grip_qpos == 0.02
    tick(native, scorer, 13, grip_contact=True, pose_error_rad=angle, **SUPPORT)
    assert not scorer.success and scorer.stable_placement_ticks == 13
    scorer.advance(scorer.last_sample)
    assert scorer.stable_placement_ticks == 13
    tick(native, scorer, grip_contact=True, pose_error_rad=angle, **SUPPORT)
    assert scorer.success
    assert scorer.last_sample.pose_error_rad == angle
    assert scorer.limits.pose_error_rad == math.radians(3)
    assert all(e["recovery"] == "diagnostic_only" for e in scorer.nonterminal_events)
    scorer.fail("gripper_orientation_violation")
    assert scorer.success and obs.reference is reference
    tick(native, scorer, 3, released=True, pose_error_rad=angle, **SUPPORT)
    assert scorer.success
    tick(native, scorer, fractured=True, pose_error_rad=angle, **SUPPORT)
    assert scorer.failure == "fracture" and not scorer.success
    original = native.ChipLifecycle()
    tick(native, original, grip_contact=True, pose_error_rad=angle)
    assert original.failure == "gripper_orientation_violation"


def test_orientation_fix_does_not_invent_grasp_or_ignore_support_loss(native):
    scorer = monitor(native).scorer
    tick(native, scorer, 30, released=True, pose_error_rad=0.3, **SUPPORT)
    assert not scorer.success and not scorer.grasp_verified
    grasp_lift(native, scorer)
    tick(native, scorer, 14, grip_contact=True, pose_error_rad=0.3, **SUPPORT)
    assert scorer.success
    tick(native, scorer, released=True, in_target=False, pose_error_rad=0.3)
    assert not scorer.success


def test_nonfinite_orientation_still_rejected(native):
    scorer = monitor(native).scorer
    with pytest.raises(RuntimeError, match="physics error"):
        tick(native, scorer, pose_error_rad=float("nan"))
    assert scorer.outcome == "invalid"


@pytest.mark.parametrize("mode,commands,seconds", [("train", 3600, 60.0), ("eval", 600, 10.0)])
def test_chip_budget_changes_eval_only_and_keeps_sixty_hz(mode, commands, seconds):
    from tactalent.roboicl.chip_evaluation import configure_evaluation_budget
    from tactalent.benchmarks.openvitac.profile import binding

    cfg = NS(
        step_lim=3600,
        final_policy_timeout_seconds=60.0,
        sim=NS(dt=1 / 120),
        decimation=1,
        policy_action_repeat=2,
    )
    spec = binding("grasp_fragile_chip")
    configure_evaluation_budget(cfg, spec, mode)
    assert cfg.step_lim == commands
    assert cfg.final_policy_timeout_seconds == seconds
    assert cfg.policy_action_repeat == 2 and cfg.sim.dt == 1 / 120
    assert spec["native_action_limit"] == 3600
    assert "evaluation_action_limit" not in binding("turn_gear_pair")


# V5 regression: a verified transport may lose pad contact over the tray.
def test_in_target_slip_preserves_grasp_lift_but_waits_for_real_support(native):
    obs = monitor(native)
    scorer = obs.scorer
    grasp_lift(native, scorer)
    reference = obs.reference
    tick(native, scorer, grip_contact=False, released=False, in_target=True, lift_m=0.02)
    assert scorer.grasp_verified and scorer.lift_verified and not scorer.success
    assert obs.reference is reference
    assert scorer.stage == "release"
    assert scorer.nonterminal_events[-1]["recovery"] == "await_stable_placement"
    tick(native, scorer, 30, in_target=True, support_contact=False, released=True)
    assert not scorer.success and scorer.stable_placement_ticks == 0
    tick(native, scorer, 13, released=True, **SUPPORT)
    assert not scorer.success
    scorer.advance(scorer.last_sample)
    assert scorer.stable_placement_ticks == 13
    tick(native, scorer, released=True, **SUPPORT)
    assert scorer.success


def test_outside_target_slip_still_requires_new_grasp_and_lift(native):
    scorer = monitor(native).scorer
    grasp_lift(native, scorer)
    tick(native, scorer, grip_contact=False, in_target=False)
    assert not scorer.grasp_verified and not scorer.lift_verified
    finish_supported(native, scorer)
    assert not scorer.success


def test_in_target_slip_before_verified_lift_cannot_count_as_placement(native):
    scorer = monitor(native).scorer
    tick(native, scorer, 6, grip_contact=True)
    tick(native, scorer, grip_contact=False, in_target=True)
    finish_supported(native, scorer)
    assert not scorer.success


@pytest.mark.parametrize(
    "changes",
    [
        dict(fractured=True),
        dict(in_target=False),
        dict(support_contact=False),
        dict(support_gap_m=0.01),
        dict(speed_m_s=0.02),
        dict(angular_speed_rad_s=0.3),
    ],
)
def test_slip_recovery_does_not_bypass_damage_or_support_limits(native, changes):
    scorer = monitor(native).scorer
    grasp_lift(native, scorer)
    tick(native, scorer, grip_contact=False, in_target=True)
    finish_supported(native, scorer, **changes)
    assert not scorer.success
