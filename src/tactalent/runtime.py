"""Isolated simulator launch, GPU locking and workspace configuration."""

import fcntl
import os
from pathlib import Path
import subprocess
import sys

BOOTSTRAP = "from PIL import Image; Image.preinit(); import runpy; runpy.run_module({module!r}, run_name='__main__')"


def benchmark_name(arguments, default="OpenViTac"):
    selected = default
    for index, argument in enumerate(arguments):
        if argument == "--benchmark" and index + 1 < len(arguments):
            selected = arguments[index + 1]
        elif argument.startswith("--benchmark="):
            selected = argument.split("=", 1)[1]
    if selected not in ("UniVTAC", "OpenViTac"):
        raise ValueError("Benchmark must be UniVTAC or OpenViTac")
    return selected


def child_environment(benchmark, gpu, *, environment=None):
    env = dict(os.environ if environment is None else environment)
    name = "ASTRA_UNIVTAC_ROOT" if benchmark == "UniVTAC" else "ASTRA_OPENVITAC_ROOT"
    if not env.get(name):
        raise ValueError("Set " + name + " to the benchmark installation")
    root = Path(env[name]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Benchmark installation does not exist: " + str(root))
    from .paths import PACKAGE_ROOT

    paths = [PACKAGE_ROOT.parent, root]
    paths += [
        root / "third_party/TacEx/source" / name
        for name in ("tacex", "tacex_assets", "tacex_tasks", "tacex_uipc")
    ]
    paths += [p for p in env.get("ASTRA_EXTRA_PYTHONPATH", "").split(os.pathsep) if p]
    paths += [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    env.setdefault("ASTRA_STORAGE_ROOT", str(Path.cwd().resolve()))
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    env.setdefault("ASTRA_STREAM", "1")
    return root, env


def launch(command, arguments):
    module = "tactalent.roboicl.run" if command == "eval" else "tactalent.roboicl.collect_training"
    if any(arg in ("-h", "--help") for arg in arguments):
        import runpy

        sys.argv = ["tactalent " + command, *arguments]
        runpy.run_module(module, run_name="__main__")
        return
    benchmark = benchmark_name(arguments)
    if command == "collect" and benchmark != "OpenViTac":
        raise ValueError("Use released training recordings for UniVTAC reference preparation")
    gpu_text = os.environ.get("GPU_ID", "0")
    if not gpu_text.isascii() or not gpu_text.isdecimal() or len(gpu_text) > 3:
        raise ValueError("GPU_ID must be an integer in [0, 999]")
    gpu = int(gpu_text)
    root, env = child_environment(benchmark, gpu)
    cmd = [sys.executable, "-B", "-c", BOOTSTRAP.format(module=module)]
    if any(arg in ("--dry-run", "--list") for arg in arguments):
        raise SystemExit(subprocess.call([*cmd, *arguments], env=env))
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), OMNI_KIT_ACCEPT_EULA="YES"
    )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = env.get(name, "4")
    env.setdefault(
        "ASTRA_RUNTIME_ROOT", str(Path(env["ASTRA_STORAGE_ROOT"]) / ".runtime" / f"gpu_{gpu}")
    )
    from .utils.storage import configure_runtime

    env.update(configure_runtime(env["ASTRA_STORAGE_ROOT"], gpu_id=gpu, environment=env))
    port = 19100 + gpu
    with Path(f"/tmp/tactalent_eval_gpu_{gpu}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("GPU already reserved") from None
        running = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if any(char.isdigit() for char in running):
            raise RuntimeError("Select an idle GPU")
        if subprocess.check_output(["ss", "-H", "-ltn", f"sport = :{port}"], text=True).strip():
            raise RuntimeError("Simulator port occupied")
        kit = (
            f"--portable-root {env['ASTRA_RUNTIME_ROOT']}/kit --/renderer/activeGpu={gpu} "
            f"--/renderer/multiGpu/enabled=false --/app/livestream/port={51000 + gpu * 10} "
            f"--/app/livestream/fixedHostPort={52000 + gpu * 10} "
            f"--/exts/omni.services.transport.server.http/port={port} "
            "--/exts/omni.services.transport.server.http/allow_port_range=false "
            + env.get("ASTRA_KIT_ARGS", "")
        )
        raise SystemExit(
            subprocess.call([*cmd, "--kit_args=" + kit, *arguments], cwd=root, env=env)
        )
