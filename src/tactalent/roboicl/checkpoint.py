"""Durable diagnostics plus exact in-process continuation after transient API errors.

Disk snapshots are NOT advertised as cold-restorable: UIPC/PhysX sensor and
task-side caches are not yet covered by a validated cross-process restore.
The paused process retains those states and resumes the pending API request.
"""

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import time
import uuid


def _json(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, default=_json)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _cpu(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return value


def _process_identity(pid):
    # starttime prevents an old checkpoint from resuming a recycled PID.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _stamp(task):
    uipc = getattr(task, "uipc_sim", None)
    return {
        "native_steps": int(task.take_action_cnt),
        "physics_steps": int(task.step_count),
        "uipc_frame": int(uipc.world.frame()) if uipc is not None else None,
    }


def capture_simulation(task, folder):
    import numpy as np
    import torch

    before = _stamp(task)
    state = _cpu(task.scene.get_state(is_relative=False))
    targets = {}
    for name, robot in task.scene.articulations.items():
        targets[name] = {
            key: _cpu(getattr(robot.data, key))
            for key in ("joint_pos_target", "joint_vel_target", "joint_effort_target")
            if hasattr(robot.data, key)
        }
    uipc = getattr(task, "uipc_sim", None)
    uipc_record = None
    if uipc is not None:
        uipc.save_frame()
        uipc_record = {
            "workspace": str(uipc.cfg.workspace),
            "frame": before["uipc_frame"],
            "format": "native libuipc world.dump",
        }
    torch.save(
        {
            "scene": state,
            "drive_targets": targets,
            "random_python": random.getstate(),
            "random_numpy": np.random.get_state(),
            "random_torch": torch.get_rng_state(),
            "random_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "counters": before,
            "uipc": uipc_record,
        },
        folder / "simulation.pt",
    )
    os.chmod(folder / "simulation.pt", 0o600)
    atomic_json(
        folder / "task.json",
        {
            "counters": before,
            "instruction": str(task.instruction),
            "metadata": getattr(task, "metadata", {}),
            "eval_success": bool(task.eval_success),
            "plan_success": bool(task.plan_success),
            "native_limit": int(task.cfg.step_lim),
        },
    )
    if before != _stamp(task):
        raise RuntimeError("Checkpoint capture advanced the simulator")
    return {
        "counters": before,
        "uipc": uipc_record,
        "cold_restart_supported": False,
        "resume_method": "same_live_process",
        "limitations": "Cross-process restoration of coupled solver, sensors and task caches is not validated.",
    }


class Checkpoints:
    def __init__(self, output, task, *, capture=capture_simulation, poll_seconds=1):
        self.output = Path(output)
        self.root = self.output / "checkpoints"
        self.root.mkdir(mode=0o700, exist_ok=False)
        self.task, self.capture, self.poll_seconds = task, capture, poll_seconds
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.current = None
        self.owner = dict(
            host=socket.gethostname(),
            pid=os.getpid(),
            process_start=_process_identity(os.getpid()),
            session=self.session,
        )
        atomic_json(self.root / "owner.json", self.owner)

    def save(self, model=None, payload=None):
        self.sequence += 1
        folder = self.root / f"request_{self.sequence:06d}"
        folder.mkdir(mode=0o700)
        receipt = self.capture(self.task, folder)
        # Explicit allowlist: never serialize an API key, environment or clients.
        context = {
            name: getattr(model, name)
            for name in (
                "messages",
                "archive",
                "ledger",
                "pending",
                "turns",
                "step",
                "fixed_prefix_length",
                "stable_prefix_hash",
                "seen_call_ids",
                "live_history_tail_start",
                "returned_reasoning_items",
                "reasoning_replay_disabled",
                "reported_tokens",
                "pending_reserved_tokens",
                "request_count",
                "local_safety_holds",
                "controller_rejection_streak",
            )
            if model is not None and hasattr(model, name)
        }
        if model is not None and getattr(model, "live_memory", None) is not None:
            context["live_memory"] = vars(model.live_memory)
        with gzip.open(folder / "dialogue.json.gz", "wt", compresslevel=1) as stream:
            json.dump({"context": context, "pending_payload": payload}, stream, default=_json)
        os.chmod(folder / "dialogue.json.gz", 0o600)
        self.current = dict(
            self.owner,
            checkpoint=str(folder),
            **receipt,
            logical_step=getattr(model, "step", None),
            decisions=getattr(model, "turns", None),
            payload_sha256=hashlib.sha256(
                json.dumps(payload, default=_json, sort_keys=True).encode()
            ).hexdigest(),
            status="ready",
            captured_at=time.time(),
        )
        atomic_json(folder / "manifest.json", self.current)
        atomic_json(self.root / "latest.json", self.current)
        # Bounded disk use for large image histories; native UIPC dumps remain in
        # the episode workspace so their internal directory layout stays intact.
        for old in sorted(self.root.glob("request_*"))[:-2]:
            if old.is_dir():
                shutil.rmtree(old)
        return self.current

    def pause(self, model, exc):
        before = _stamp(self.task)
        token = uuid.uuid4().hex
        state = dict(
            self.current,
            status="paused_api",
            pause_token=token,
            paused_at=time.time(),
            error_code=getattr(exc, "error_code", "transport_error"),
            error_type=type(exc).__name__,
            counters=before,
            resume_requires_same_process=True,
        )
        atomic_json(self.root / "latest.json", state)
        atomic_json(
            self.output / "progress.json",
            dict(
                status="paused_api",
                native_steps=before["native_steps"],
                decisions=model.turns,
                checkpoint=state["checkpoint"],
                resume_command=f"python -m tactalent.roboicl.checkpoint --resume {self.output}",
                note="Simulator and policy retained; no physics steps or API requests while paused.",
            ),
        )
        model._log(
            "checkpoint_recovery.jsonl",
            {
                "event": "paused_api",
                "native_steps": before["native_steps"],
                "error_code": state["error_code"],
                "pause_token": token,
                "robot_action_replayed": False,
            },
        )
        control = self.root / "control.json"
        while True:
            if control.exists():
                command = json.loads(control.read_text())
                control.unlink()
                if command.get("session") != self.session or command.get("pause_token") != token:
                    continue
                action = command.get("action")
                if action == "stop":
                    raise RuntimeError("Paused API evaluation stopped by operator") from exc
                if action != "resume":
                    continue
                if _stamp(self.task) != before:
                    raise RuntimeError("Paused simulation advanced; refusing continuation")
                model._log(
                    "checkpoint_recovery.jsonl",
                    {
                        "event": "resumed_api",
                        "native_steps": before["native_steps"],
                        "pause_token": token,
                        "robot_action_replayed": False,
                        "pending_reserved_tokens": getattr(model, "pending_reserved_tokens", None),
                    },
                )
                atomic_json(
                    self.root / "latest.json",
                    dict(state, status="resuming", resumed_at=time.time()),
                )
                atomic_json(
                    self.output / "progress.json",
                    dict(
                        status="running",
                        native_steps=before["native_steps"],
                        decisions=model.turns,
                        resumed_from=state["checkpoint"],
                    ),
                )
                return
            time.sleep(self.poll_seconds)

    def install(self, model):
        from roboicl.policy.bounded_policy import RequestTransportError

        original = model._request

        def request(payload):
            self.save(model, payload)
            while True:
                try:
                    response = original(payload)
                    atomic_json(
                        self.root / "latest.json", dict(self.current, status="response_received")
                    )
                    return response
                except RequestTransportError as exc:
                    # Deterministic payload repairs keep their official bounded
                    # retry rules; only transient exhausted retries may pause.
                    if not exc.retryable or getattr(exc, "recovery", None) is not None:
                        raise
                    self.save(model, payload)
                    self.pause(model, exc)
                    # Keep all attempted-request reservations and usage. Only
                    # clear the failure latch, as the upstream retry path does.
                    model.budget_poisoned = False

        model._request = request

    def close(self, status):
        if self.current:
            atomic_json(
                self.root / "latest.json",
                dict(
                    self.current,
                    status="closed",
                    evaluation_status=status,
                    live_resume_available=False,
                ),
            )


def send_control(output, action):
    root = Path(output) / "checkpoints"
    state = json.loads((root / "latest.json").read_text())
    if state.get("status") != "paused_api":
        raise ValueError("Episode is not paused awaiting API continuation")
    if state.get("host") != socket.gethostname():
        raise ValueError("Run this command on the checkpoint's execution host: " + state["host"])
    try:
        identity = _process_identity(int(state["pid"]))
    except (OSError, ValueError):
        raise ValueError(
            "Original simulator process is gone; cold restart is not supported"
        ) from None
    if identity != state["process_start"]:
        raise ValueError("Original simulator PID was recycled; refusing continuation")
    atomic_json(
        root / "control.json",
        dict(
            action=action,
            session=state["session"],
            pause_token=state["pause_token"],
            requested_at=time.time(),
        ),
    )
    return {k: state[k] for k in ("host", "pid", "checkpoint", "counters")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resume", type=Path)
    group.add_argument("--stop", type=Path)
    args = parser.parse_args()
    print(json.dumps(send_control(args.resume or args.stop, "resume" if args.resume else "stop")))


if __name__ == "__main__":
    main()
