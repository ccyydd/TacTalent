import gzip
import json
import threading
import time
from types import SimpleNamespace
import pytest
from tactalent.roboicl.checkpoint import Checkpoints, atomic_json, send_control


def make(tmp_path):
    task = SimpleNamespace(take_action_cnt=20, step_count=40, uipc_sim=None)

    def capture(task, folder):
        (folder / "simulation.test").write_text("snapshot")
        return {
            "counters": {
                "native_steps": task.take_action_cnt,
                "physics_steps": task.step_count,
                "uipc_frame": None,
            },
            "cold_restart_supported": False,
            "resume_method": "same_live_process",
        }

    cp = Checkpoints(tmp_path, task, capture=capture, poll_seconds=0.005)
    model = SimpleNamespace(
        turns=4,
        step=20,
        api_key="NEVER_SERIALIZE_THIS",
        messages=[{"role": "user", "content": "observation"}],
        archive={20: {"state": [1, 2]}},
        reported_tokens=10,
        pending_reserved_tokens=12,
        budget_poisoned=True,
        _log=lambda *args: None,
    )
    return task, cp, model


def wait_paused(cp):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        p = cp.root / "latest.json"
        if p.exists() and json.loads(p.read_text()).get("status") == "paused_api":
            return
        time.sleep(0.005)
    raise AssertionError("did not pause")


def test_checkpoint_is_private_and_retains_only_two(tmp_path):
    _, cp, model = make(tmp_path)
    for _ in range(3):
        cp.save(model, {"input": [1]})
    assert len(list(cp.root.glob("request_*"))) == 2
    latest = json.loads((cp.root / "latest.json").read_text())
    text = gzip.open(cp.root / "request_000003/dialogue.json.gz", "rt").read()
    assert "NEVER_SERIALIZE_THIS" not in text
    assert json.loads(text)["context"]["pending_reserved_tokens"] == 12
    assert latest["cold_restart_supported"] is False
    assert cp.root.stat().st_mode & 0o077 == 0


def test_transient_failure_resumes_same_request_without_actions(tmp_path):
    from tactalent.roboicl.upstream import VENDOR
    from roboicl.policy.bounded_policy import RequestTransportError

    task, cp, model = make(tmp_path)
    calls = []

    def request(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise RequestTransportError(
                "no capacity", retryable=True, error_code="policy_rpc_error"
            )
        assert not model.budget_poisoned
        return {"accepted": True}

    model._request = request
    cp.install(model)
    result = []
    errors = []

    def run():
        try:
            result.append(model._request({"input": ["CURRENT"]}))
        except Exception as e:
            errors.append(e)

    worker = threading.Thread(target=run)
    worker.start()
    wait_paused(cp)
    assert len(calls) == 1
    assert (task.take_action_cnt, task.step_count) == (20, 40)
    send_control(tmp_path, "resume")
    worker.join(3)
    assert not worker.is_alive() and not errors
    assert result == [{"accepted": True}] and calls[0] == calls[1]
    assert (task.take_action_cnt, task.step_count) == (20, 40)
    assert model.pending_reserved_tokens == 12 and model.reported_tokens == 10


def test_nonretryable_failure_is_not_paused(tmp_path):
    from tactalent.roboicl.upstream import VENDOR
    from roboicl.policy.bounded_policy import RequestTransportError

    _, cp, model = make(tmp_path)

    def request(payload):
        raise RequestTransportError("bad key", retryable=False)

    model._request = request
    cp.install(model)
    with pytest.raises(RequestTransportError):
        model._request({})
    assert json.loads((cp.root / "latest.json").read_text())["status"] == "ready"


def test_changed_simulation_refuses_resume(tmp_path):
    task, cp, model = make(tmp_path)
    cp.save(model, {})
    errors = []

    def run():
        try:
            cp.pause(model, RuntimeError())
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    wait_paused(cp)
    task.step_count += 1
    send_control(tmp_path, "resume")
    worker.join(3)
    assert errors and "advanced" in str(errors[0])


def test_dead_process_cannot_be_cold_resumed(tmp_path):
    _, cp, model = make(tmp_path)
    cp.save(model, {})
    state = json.loads((cp.root / "latest.json").read_text())
    state.update(status="paused_api", pid=2147483647, pause_token="token")
    atomic_json(cp.root / "latest.json", state)
    with pytest.raises(ValueError, match="cold restart"):
        send_control(tmp_path, "resume")


def test_operator_stop_exits_pause(tmp_path):
    _, cp, model = make(tmp_path)
    cp.save(model, {})
    errors = []

    def run():
        try:
            cp.pause(model, RuntimeError())
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    wait_paused(cp)
    send_control(tmp_path, "stop")
    worker.join(3)
    assert errors and "stopped by operator" in str(errors[0])
