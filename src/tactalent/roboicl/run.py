"""Run the official RoboICL dialogue on one explicit benchmark task and held-out seed."""

import argparse, hashlib, json, math, os, threading, time, traceback, urllib.request
from pathlib import Path
import numpy as np
from .upstream import identity, protocol_sha256
from .benchmarks import (
    binding,
    all_bindings,
    validate_reference_horizon,
    task_context,
    task_prompt,
    validate_reference_task_context,
)
from .environment import create, read
from .demonstrations import validate_reference_demonstrations
from .observations import (
    ICL_MODES,
    resolve_mode,
    validate_reference_inputs,
    input_contract,
    image_names,
    panel,
)


def write(path, value):
    def convert(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    Path(path).write_text(json.dumps(value, indent=2, default=convert))


def preflight(args):
    spec = binding(args.benchmark, args.task, args.variant)
    args.icl_mode = resolve_mode(spec, getattr(args, "icl_mode", None))
    reference = None
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference.get("binding") != spec or reference.get("protocol_sha256") != protocol_sha256(
            spec
        ):
            raise ValueError("Reference task/prompt/source version mismatch; prepare again")
        validate_reference_horizon(reference, spec)
        validate_reference_task_context(reference, spec)
        validate_reference_inputs(reference, spec, args.icl_mode)
        validate_reference_demonstrations(reference, spec)
        if reference["action_repeat"] != 2 or not math.isclose(reference["frequency"], 60):
            raise ValueError("Expected the declared 60 Hz recorded demonstration")
    if args.backend == "live" and reference is None:
        raise ValueError("Live evaluation requires a current task-matched reference")
    from .evaluation import validate_episode

    validate_episode(spec, args.seed, getattr(args, "evaluation_case", None), reference)
    return spec, reference


def attach_usage_logger(model, output):
    records = []
    original = model._log

    def log(name, value):
        original(name, value)
        row = None
        if name == "http_transport.jsonl" and value.get("event") in (
            "http_error",
            "transport_error",
        ):
            row = {
                "api_provider": "gateway",
                "request": value.get("request"),
                "usage": {},
                "transport_error": value.get("http_status", value.get("exception_type")),
                "cost_status": "unknown_no_usage_returned",
            }
        if name == "budget.jsonl" and value.get("event") == "response":
            u = value["usage"]
            usage = {
                "prompt_tokens": u.get("input_tokens"),
                "completion_tokens": u.get("output_tokens"),
                "total_tokens": u.get(
                    "total_tokens", u.get("input_tokens", 0) + u.get("output_tokens", 0)
                ),
                "prompt_tokens_details": u.get("input_tokens_details", {}),
                "completion_tokens_details": u.get("output_tokens_details", {}),
            }
            usage.update({k: u[k] for k in ("cost", "currency", "cost_details") if k in u})
            row = {"api_provider": "gateway", "usage": usage, "request": value["request"]}
        if row:
            from tactalent.utils.usage import summarize_attempts

            records.append(row)
            write(
                output / "api_usage.json",
                {"attempts": records, "summary": summarize_attempts(records)},
            )

    model._log = log


def native_stopped(task, execution):
    return bool(
        task.eval_success
        or task.take_action_cnt >= task.cfg.step_lim
        or not task.plan_success
        or task.check_early_stop()
        or execution.get("truncated", False)
    )


def run(args, spec, reference):
    from tactalent.utils.storage import require_output_path

    output = require_output_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    app = task = model = checkpoints = None
    icl_mode = resolve_mode(spec, getattr(args, "icl_mode", None))
    started = time.monotonic()
    result = {
        "status": "error",
        "benchmark": spec["benchmark"],
        "task": spec["task"],
        "variant": spec["variant"],
        "seed": args.seed,
        "backend": args.backend,
        "method": "official_roboicl_native_binding",
        "upstream": identity(),
        "icl_mode": icl_mode,
    }
    from .evaluation import validate_episode

    evaluation_case = getattr(args, "evaluation_case", None)
    reviewed_case = validate_episode(spec, args.seed, evaluation_case, reference)
    result["evaluation_case"] = reviewed_case
    phase = "initialization"
    try:
        app, task = create(
            output, args.seed, "eval", args.kit_args, spec, evaluation_case=evaluation_case
        )
        from .controller import Controller, command_budget, native_action_cost
        from .calibration import verify

        repeat = reference["action_repeat"] if reference else 2
        controller = Controller(task, repeat)
        calibration = verify(task, spec)
        state, images = read(task, spec, icl_mode)
        initial_native_steps = int(task.take_action_cnt)
        initial_physics_steps = int(task.step_count)
        maximum = command_budget(task, repeat)
        native_limit = int(task.cfg.step_lim)
        if native_action_cost(task, repeat) != spec["native_actions_per_command"]:
            raise ValueError("Native action accounting differs from the reviewed binding")
        sizes = {n: [im.shape[1], im.shape[0]] for n, im in images.items()}
        runtime = {
            "binding": spec,
            "native_horizon": native_limit,
            "evaluation_case": reviewed_case,
            "task_context": task_context(spec),
            "task_prompt": task_prompt(spec, task.instruction),
            "native_instruction": str(task.instruction),
            "action_repeat": repeat,
            "native_actions_per_command": native_action_cost(task, repeat),
            "max_commands": maximum,
            "control_frequency_hz": 1 / (float(task.cfg.sim.dt) * repeat),
            "max_policy_physics_steps": maximum * repeat,
            "native_dt": float(task.cfg.sim.dt),
            "initial_native_steps": initial_native_steps,
            "fk": controller.fk_audit,
            "head_calibration": calibration,
            "model_images": list(image_names(spec, icl_mode)),
            "model_cameras": spec["cameras"],
            "icl_mode": icl_mode,
            "input_contract": input_contract(spec, icl_mode),
            "model_tactile": icl_mode == "vision_tactile",
            "native_source_sizes_wh": sizes,
            "panel_size_wh": list(panel(images, spec["cameras"], icl_mode, spec)[0].size),
            "protocol_sha256": protocol_sha256(spec),
        }
        if reference:
            runtime["demonstration_contract"] = reference["demonstration_contract"]
            runtime["predict_horizon"] = runtime["execute_horizon"] = validate_reference_horizon(
                reference, spec
            )
            runtime["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
            (output / "reference.snapshot.json").write_bytes(args.reference.read_bytes())
        if "evaluation_rules" in task.metadata:
            rules = dict(task.metadata["evaluation_rules"])
            runtime["evaluation_rules"] = rules
            result["evaluation_rules"] = rules
            result["method"] = "official_roboicl_modified_chip_protocol"
        write(output / "runtime.json", runtime)
        phase = "execution"
        from .checkpoint import Checkpoints

        checkpoints = Checkpoints(output, task)
        logical_step = 0
        if args.backend == "smoke":
            from PIL import Image

            for name, img in images.items():
                Image.fromarray(img).save(output / ("initial_" + name + ".png"))
            checkpoints.save()
            receipts = []
            # An observed-state hold and a 0.2 mm IK probe; no model/API call.
            commands = [
                {
                    "mode": "joint",
                    "arm": state["arm_joint_state"],
                    "gripper": state["ee_joint_state"][0],
                },
                {
                    "mode": "ee",
                    "ee_pose": [*state["ee_pose"]],
                    "gripper": state["ee_joint_state"][0],
                },
            ]
            commands[1]["ee_pose"][2] += 0.0002
            for command in commands:
                if native_stopped(task, controller.last_result):
                    break
                qpos, error = controller.plan(command)
                if error:
                    raise RuntimeError("Smoke IK probe failed: " + json.dumps(error))
                if not controller.apply(qpos, command):
                    raise RuntimeError("Native smoke action failed")
                logical_step += 1
                state, images = read(task, spec, icl_mode)
                receipts.append(
                    dict(
                        command=command,
                        tracking=controller.tracking(command, state),
                        execution=controller.last_result,
                    )
                )
            if not receipts:
                raise RuntimeError("Native reset left no executable action")
            checkpoints.save()
            write(output / "smoke_receipts.json", receipts)
            result.update(
                status="smoke_passed",
                termination="integration_check",
                formal_evaluation=False,
                api_calls=0,
                commands=logical_step,
            )
        else:
            from .policy import Model, ProxyHTTPS
            from roboicl.policy.action_feedback import TrackingGuard

            urllib.request.install_opener(
                urllib.request.build_opener(urllib.request.ProxyHandler({}), ProxyHTTPS())
            )
            if not os.environ.get("ASTRA_API_KEY"):
                raise ValueError("Set ASTRA_API_KEY in the process environment")
            model = Model(args.reference, output, icl_mode)
            attach_usage_logger(model, output)
            model.live_calibration = calibration
            model.update_native(state, images, 0, str(task.instruction), maximum)
            (output / "prompt.txt").write_text(model.prompt)
            checkpoints.install(model)
            while not native_stopped(task, controller.last_result):
                commands = model.get_action()
                guard = TrackingGuard()
                for i, command in enumerate(commands):
                    if native_stopped(task, controller.last_result):
                        break
                    qpos, error = controller.plan(
                        command, model.dialogue_config.max_joint_increment_rad
                    )
                    if error:
                        if command["mode"] == "joint":
                            raise RuntimeError("Safety hold cannot fail IK")
                        event = dict(
                            type="action_rejected_ik_continuity",
                            rejected_action_executed=False,
                            action_index=i,
                            discarded_commands=len(commands) - i,
                            **error,
                        )
                        model.notify_action_rejection({"step": model.step, "event": event})
                        break
                    ok = controller.apply(qpos, command)
                    logical_step += 1
                    state, images = read(task, spec, icl_mode)
                    tracking = controller.tracking(command, state)
                    model.update_native(
                        state,
                        images,
                        logical_step,
                        str(task.instruction),
                        maximum,
                        command,
                        tracking,
                    )
                    model.ledger[-1]["native_execution"] = dict(
                        controller.last_result, native_count=int(task.take_action_cnt)
                    )
                    if not ok:
                        raise RuntimeError("Native qpos execution failed")
                    if native_stopped(task, controller.last_result):
                        break
                    if guard.observe(tracking):
                        model.ledger[-1]["controller_event"] = dict(
                            type="chunk_interrupted_tracking_error",
                            discarded_commands=len(commands) - i - 1,
                            reason="Two consecutive large target residuals.",
                        )
                        break
                write(
                    output / "progress.json",
                    dict(
                        status="running",
                        decisions=model.turns,
                        native_steps=int(task.take_action_cnt),
                        commands=logical_step,
                        max_steps=native_limit,
                        elapsed_s=time.monotonic() - started,
                    ),
                )
            model.finalize()
            result.update(
                status="success" if task.eval_success else "failed",
                termination=(
                    "native_success"
                    if task.eval_success
                    else (
                        "native_horizon"
                        if int(task.take_action_cnt) >= native_limit
                        else "native_early_stop"
                    )
                ),
                commands=logical_step,
                formal_evaluation=True,
            )
        result.update(
            native_success=bool(task.eval_success),
            native_steps=int(task.take_action_cnt),
            policy_native_action_delta=int(task.take_action_cnt) - initial_native_steps,
            policy_physics_step_delta=int(task.step_count) - initial_physics_steps,
        )
        if hasattr(task, "_get_gear_pair_turn_result"):
            result["geometry"] = task._get_gear_pair_turn_result()
        # Native task metrics are written only to evaluation output, never to Model.
        if hasattr(task, "metadata"):
            result["native_diagnostics"] = {
                k: task.metadata[k]
                for k in ("success_diagnostics", "early_stop", "physical_result")
                if k in task.metadata
            }
    except Exception as exc:
        app = getattr(exc, "_roboicl_app", app)
        task = getattr(exc, "_roboicl_task", task)
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            termination=(
                "initialization_error" if phase == "initialization" else "integration_error"
            ),
        )
        (output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if checkpoints is not None:
            checkpoints.close(result["status"])
        if model is not None:
            result.update(
                decisions=model.turns,
                reported_tokens=model.reported_tokens,
                trace_dir=str(model.episode_dir),
            )
        if task is not None:
            result["native_steps"] = int(task.take_action_cnt)
            try:
                task.clean_cache(mean_steps=task.cfg.step_lim, result=result["status"])
            except Exception as exc:
                result["cleanup_error"] = str(exc)
        if task is not None and "evaluation_rules" in getattr(task, "metadata", {}):
            result["evaluation_rules"] = dict(task.metadata["evaluation_rules"])
            result["success_contract"] = task.metadata.get("success_contract")
            # Cleanup finalizes incomplete lifecycles; record the final verdict.
            result["native_diagnostics"] = {
                k: task.metadata[k]
                for k in ("success_diagnostics", "early_stop", "physical_result", "terminal_reason")
                if k in task.metadata
            }
        result["elapsed_s"] = time.monotonic() - started
        write(output / "result.json", result)
        write(
            output / "progress.json",
            {
                k: result.get(k)
                for k in ("status", "termination", "native_steps", "decisions", "elapsed_s")
            },
        )
        guard = threading.Timer(25, lambda: os._exit(0 if result["status"] != "error" else 1))
        guard.daemon = True
        guard.start()
        if task is not None:
            task.close()
        if app is not None:
            app.close()
        guard.cancel()
    return 1 if result["status"] == "error" else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["UniVTAC", "OpenViTac"], default="OpenViTac")
    p.add_argument("--task", default="turn_gear_pair")
    p.add_argument("--variant")
    p.add_argument("--evaluation-case", help="Reviewed full-evaluation condition ID")
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=1000000)
    p.add_argument("--kit_args", default="")
    p.add_argument("--icl-mode", choices=ICL_MODES, help="Default: benchmark vision_tactile")
    p.add_argument("--backend", choices=["live", "smoke"], default="live")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        print(json.dumps(all_bindings(), indent=2))
        return
    spec, reference = preflight(a)
    if a.dry_run:
        print(
            json.dumps(
                {
                    "binding": spec,
                    "reference_validated": reference is not None,
                    "backend": a.backend,
                    "seed": a.seed,
                    "icl_mode": a.icl_mode,
                    "evaluation_case": a.evaluation_case,
                },
                indent=2,
            )
        )
        return
    if a.output is None:
        p.error("--output is required")
    raise SystemExit(run(a, spec, reference))


if __name__ == "__main__":
    main()
