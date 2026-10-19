"""Exercise the pinned native damage path, without launching Isaac or calling an API."""

import ast
import numpy as np
import pytest
from types import SimpleNamespace as NS
from tactalent.benchmarks.openvitac.profile import ROOT
from tactalent.roboicl.chip_evaluation import (
    DiagnosticImpactTracker,
    PROTOCOL,
    evaluation_task_class,
    evaluation_rules,
)


@pytest.fixture
def damage_task():
    path = ROOT / "envs/grasp_fragile_chip.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if getattr(n, "name", None) in {"impact_breaks_chip", "SupportImpactTracker"}
    ]
    task = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Task")
    step = next(n for n in task.body if getattr(n, "name", "") == "_advance_chip_physics")

    # Keep the actual native damage step, including super()._step, velocity
    # history, diagnostic landing record, and the subsequent strain branch.
    class Physics:
        def _step(self, is_save=True):
            self.step_count += 1

    shell = ast.ClassDef(
        name="NativeDamage",
        bases=[ast.Name(id="Physics", ctx=ast.Load())],
        keywords=[],
        body=[step],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=nodes + [shell], type_ignores=[]))
    ns = dict(
        np=np,
        Physics=Physics,
        capture_geometry_state=lambda sim: None,
        TABLE_TOP_Z=0.0,
        CHIP_IMPACT_SPEED_M_S=0.09,
    )
    exec(compile(module, str(path), "exec"), ns)

    class NativeTask(ns["NativeDamage"]):
        def __init__(self, strain=0.0):
            self.reset()
            self.strain = strain

        def reset(self):
            self.metadata = {}
            self.fractured = False
            self.monitor_chip_contact = False
            self._chip_damage_armed = True
            self._fragment_release_pending = False
            self.uipc_sim = None
            self.step_count = 7
            self.cfg = NS(
                sim=NS(dt=1 / 120), decimation=1, step_lim=3600, final_policy_timeout_seconds=60
            )
            self.active_chip = NS(vertices=np.array([[0.0, 0.0, 0.00633]]))
            self.tray = NS(get_pose=lambda: NS(p=np.zeros(3)))
            self._chip_previous_center = np.array([0.0, 0.0, 0.01])
            self._chip_previous_velocity_z = -0.25
            self._chip_impact_tracker = ns["SupportImpactTracker"](0.09)
            self._chip_impact_tracker.advance(0.0, False, 0.01)
            self._chip_strain_baseline = 0.0
            self.episode = NS(fracture_strain=0.0008)
            self.tactile_depth_trace = []
            self.tactile_strain_trace = []
            self.strain_checked = False

        def _plate_support_resultant(self):
            return {"support_force_N": 0.07}

        def _ensure_chip_strain_reference(self):
            pass

        def _read_tactile_depth(self):
            return np.array([34.0, 34.0])

        def _chip_strain(self):
            self.strain_checked = True
            return self.strain

        def _fracture_chip(self, depths, strain=None, *, reason="squeeze", impact_speed=None):
            self.fractured = True
            self.metadata["fracture_reason"] = reason

    # Monitor is needed by the adapter reset; use the real lifecycle fixture
    # from the other regression module to avoid initializing Isaac here.
    from test_roboicl_chip_evaluation import native as lifecycle_fixture

    lifecycle = lifecycle_fixture.__wrapped__()

    class Monitor:
        def __init__(self, task):
            self.task = task
            self.scorer = lifecycle.ChipLifecycle()

    return NS(
        Native=NativeTask,
        Eval=evaluation_task_class(NativeTask, Monitor),
        Tracker=ns["SupportImpactTracker"],
    )


@pytest.mark.parametrize(
    "speed", [0.1914578, 0.2497384, 0.4077679, 0.1632070, 0.3422013, 0.1442385]
)
def test_recorded_failed_landing_speeds_no_longer_break_chip(damage_task, speed):
    original = damage_task.Native()
    adapted = damage_task.Eval()
    for task in (original, adapted):
        task._chip_previous_velocity_z = -speed
        task._advance_chip_physics()
    assert original.fractured and original.metadata["fracture_reason"] == "impact"
    assert not adapted.fractured
    assert adapted.strain_checked
    assert np.array_equal(adapted._chip_previous_center, adapted.active_chip.vertices.mean(0))
    assert adapted._chip_previous_velocity_z != -speed
    landing = adapted.metadata["physical_landing_events"][-1]
    assert landing["incoming_speed_m_s"] == speed and not landing["fractures"]
    event = adapted.metadata["impact_threshold_events"][-1]
    assert event["incoming_speed_m_s"] == speed and not event["fracture_applied"]
    assert event["protocol"] == PROTOCOL
    assert not adapted._chip_impact_tracker.flight_armed


def test_same_step_strain_fracture_is_preserved(damage_task):
    task = damage_task.Eval(strain=0.001)
    task._advance_chip_physics()
    assert task.fractured and task.metadata["fracture_reason"] == "squeeze"
    assert task.strain_checked


def test_slow_successful_landing_and_supported_settling_remain_unmodified(damage_task):
    task = damage_task.Eval()
    task._chip_previous_velocity_z = -0.0600892
    task._advance_chip_physics()
    assert not task.fractured and not task.metadata.get("impact_threshold_events")
    task._chip_previous_velocity_z = -0.3
    task._advance_chip_physics()
    assert not task.fractured and not task.metadata.get("impact_threshold_events")
    assert len(task.metadata["physical_landing_events"]) == 1


def test_new_flight_records_new_event_and_reset_replaces_wrapper(damage_task):
    task = damage_task.Eval()
    task._advance_chip_physics()
    first = task._chip_impact_tracker
    first.advance(0.0, False, 0.01)
    task._chip_previous_velocity_z = -0.2
    task._advance_chip_physics()
    assert len(task.metadata["impact_threshold_events"]) == 2
    task.reset()
    task._advance_chip_physics()
    assert task._chip_impact_tracker is not first
    assert len(task.metadata["impact_threshold_events"]) == 1
    assert evaluation_rules(task)["impact_fracture_enforced"] is False


def test_invalid_contact_data_still_raises(damage_task):
    task = damage_task.Eval()
    tracker = DiagnosticImpactTracker(damage_task.Tracker(0.09), task)
    with pytest.raises(ValueError):
        tracker.advance(float("nan"), True, 0.0003)
