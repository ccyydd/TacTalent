"""Regression for accidentally applying evaluation handoff to expert collection."""

from types import SimpleNamespace
import pytest
from tactalent.roboicl.native_settings import apply_settings


def config():
    return SimpleNamespace(
        skip_pre_move=False,
        eval_start_delay_steps=20,
        save_pre_move=False,
        decimation=1,
        reset_final_steps=30,
        step_lim=400,
    )


def test_native_collection_keeps_pre_move_despite_eval_yaml():
    cfg = config()
    apply_settings(
        cfg,
        dict(
            skip_pre_move=True,
            eval_start_delay_steps=0,
            save_pre_move=True,
            save_frequency=2,
            step_lim=999,
            reset_final_steps=40,
        ),
        benchmark="OpenViTac",
        mode="collect",
    )
    assert cfg.skip_pre_move is False
    assert cfg.eval_start_delay_steps == 20
    assert cfg.save_pre_move is True and cfg.save_frequency == 2
    assert cfg.reset_final_steps == 40
    assert cfg.step_lim == 400


@pytest.mark.parametrize("skip,delay", [(True, 0), (False, 20)])
def test_evaluation_handoff_is_preserved(skip, delay):
    cfg = config()
    apply_settings(cfg, {"skip_pre_move": skip}, benchmark="OpenViTac", mode="eval")
    assert cfg.skip_pre_move == skip
    assert cfg.eval_start_delay_steps == delay


@pytest.mark.parametrize("mode", ["collect", "eval"])
def test_univtac_settings_unchanged(mode):
    cfg = config()
    apply_settings(
        cfg,
        dict(skip_pre_move=True, save_pre_move=True, step_lim=300),
        benchmark="UniVTAC",
        mode=mode,
    )
    assert cfg.skip_pre_move is True and cfg.save_pre_move is True
    assert cfg.step_lim == 300
    assert cfg.eval_start_delay_steps == 20


def test_collection_preserves_task_default_instead_of_forcing_false():
    cfg = config()
    cfg.skip_pre_move = True
    apply_settings(cfg, {"skip_pre_move": False}, benchmark="OpenViTac", mode="collect")
    assert cfg.skip_pre_move is True
