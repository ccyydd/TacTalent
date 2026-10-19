"""The chip adapter repairs reset geometry, never policy-time contact."""

from types import SimpleNamespace
import numpy as np
import pytest
from tactalent.roboicl import openvitac_chip


def fake_task(points, targets, ids=(0, 1)):
    attachment = SimpleNamespace(
        attachment_points_idx=list(ids), _compute_aim_positions=lambda: np.asarray(targets)
    )
    geometry = SimpleNamespace(positions=lambda: SimpleNamespace(view=lambda: np.asarray(points)))
    sensor = SimpleNamespace(
        attachment=attachment,
        gelpad=SimpleNamespace(geo_slot_list=[SimpleNamespace(geometry=lambda: geometry)]),
    )
    return SimpleNamespace(_tactile_manager=SimpleNamespace(tactiles={"left": sensor}))


def test_detached_pad_is_detected_even_with_valid_render_vertices():
    task = fake_task([[0.2, 0, 0], [0.2, 1, 0]], [[0, 0, 0], [0, 1, 0]])
    assert openvitac_chip.attachment_residuals(task)["left"]["max_error_m"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "points,targets,ids",
    [
        ([[float("nan"), 0, 0]], [[0, 0, 0]], (0,)),
        ([[0, 0, 0]], [[float("inf"), 0, 0]], (0,)),
        ([[0, 0, 0]], [], ()),
    ],
)
def test_invalid_attachment_is_an_initialization_error(points, targets, ids):
    with pytest.raises(RuntimeError, match="invalid_physics"):
        openvitac_chip.attachment_residuals(fake_task(points, targets, ids))


def test_reset_alignment_precedes_native_rebuild_and_preserves_return(monkeypatch):
    events = []

    class Native:
        metadata = {}

        def _reset_actors(self):
            events.append("native_rebuild")

        def reset(self):
            self._reset_actors()
            events.append("native_settle")
            return "native_observation"

    monkeypatch.setattr(
        openvitac_chip,
        "align_unloaded_pads",
        lambda task: events.append("align") or {"left": {"max_reset_shift_m": 0.2}},
    )
    monkeypatch.setattr(
        openvitac_chip, "attachment_residuals", lambda task: {"left": {"max_error_m": 1e-6}}
    )
    task = openvitac_chip.task_class(Native)()
    assert task.reset() == "native_observation"
    assert events == ["align", "native_rebuild", "native_settle"]


def test_failed_startup_alignment_cannot_become_policy_failure(monkeypatch):
    class Native:
        metadata = {}

        def reset(self):
            return "unusable_observation"

    monkeypatch.setattr(
        openvitac_chip, "attachment_residuals", lambda task: {"left": {"max_error_m": 0.2}}
    )
    with pytest.raises(RuntimeError, match="invalid_physics.*detached"):
        openvitac_chip.task_class(Native)().reset()
