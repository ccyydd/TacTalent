"""Keep outputs, simulator state and caches in the configured workspace."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from tactalent.paths import workspace_root
from collections.abc import Mapping, Sequence
from pathlib import Path

OUTPUT_ROOT = Path(os.environ.get("ASTRA_STORAGE_ROOT", str(workspace_root())))


class StoragePathError(ValueError):
    """A requested write location is outside the permitted storage mount."""


def require_output_path(path: str | os.PathLike[str]) -> Path:
    """Resolve and validate a write location without creating any files.

    Existing symbolic links are followed before checking the mount boundary,
    including links in a parent of a not-yet-created output path. Callers must
    use the returned canonical path for subsequent writes. This checks the
    current filesystem; it does not prevent another process from changing
    symbolic links after validation.

    Args:
        path: An absolute or relative output location.

    Returns:
        The absolute location inside the configured workspace.

    Raises:
        StoragePathError: The path is invalid, cannot be resolved, or resolves
            outside the fixed configured workspace mount.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise StoragePathError("A write path must be a string or PathLike.")
    if isinstance(path, str) and not path:
        raise StoragePathError("A write path must not be empty.")
    try:
        requested = Path(path)
        resolved = requested.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StoragePathError("The write path could not be resolved.") from exc

    # Do not resolve the mount itself: a mount-name symlink to another disk
    # must not turn that other disk into an allowed location.
    mount = Path(os.path.abspath(OUTPUT_ROOT))
    try:
        resolved.relative_to(mount)
    except ValueError as exc:
        raise StoragePathError(
            f"Write path must resolve inside {mount}; resolved to {resolved}."
        ) from exc
    return resolved


def _exportable_path(path: str | os.PathLike[str]) -> Path:
    """Validate a path that will also be printed in a line-based protocol."""
    resolved = require_output_path(path)
    if any(character in str(resolved) for character in ("\r", "\n", "\x00")):
        raise StoragePathError("Runtime paths must not contain line breaks or NUL.")
    return resolved


def _check_directory(path: Path) -> None:
    """Reject existing files in a directory path before any creation occurs."""
    for current in (path, *path.parents):
        if current.exists() and not current.is_dir():
            raise StoragePathError(f"Runtime directory contains a file: {current}.")


def _check_file(path: Path) -> None:
    """Check a writable file target and its parent without reading its contents."""
    _check_directory(path.parent)
    if path.exists() and not path.is_file():
        raise StoragePathError(f"Runtime write target must be a file: {path}.")


def _validate_existing_symlinks(root: Path) -> None:
    """Inspect existing extraction paths, including configured workspace directory-link targets."""
    if not root.exists():
        return
    pending = [root]
    visited = set()
    while pending:
        directory = _exportable_path(pending.pop())
        if directory in visited:
            continue
        visited.add(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    destination = _exportable_path(entry.path)
                    if destination.is_dir():
                        pending.append(destination)
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))


def configure_runtime(
    package_root: str | os.PathLike[str] | None = None,
    *,
    gpu_id: str | int | None = None,
    environment: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Configure project runtime directories after validating every location.

    Only the four ASTRA storage overrides, PACKAGE_ROOT and GPU_ID are read
    from the environment. Existing library-specific cache overrides are
    replaced in the returned mapping. Credentials are neither read nor
    changed. Callers must export the returned mapping before importing runtime
    libraries. No directories, weights or configuration files are created
    when ``dry_run`` is true, and no weights are downloaded in either mode.

    Args:
        package_root: Evaluation project directory. Defaults to PACKAGE_ROOT
            from the environment, or the installed module's project directory.
        gpu_id: Physical GPU index, used for the default runtime directory.
        environment: Environment to read storage roots from, without mutation.
        dry_run: Validate and describe all locations without writing anything.

    Returns:
        A mapping of safe environment variable names to canonical configured workspace paths,
        plus the disabled Omniverse Hub mode.

    Raises:
        StoragePathError: Any root, derived path, or configuration target is
            invalid or outside configured workspace. No locations are created in this case.
    """
    source = os.environ if environment is None else environment
    selected_gpu = source.get("GPU_ID", "0") if gpu_id is None else gpu_id
    if isinstance(selected_gpu, bool) or not str(selected_gpu).isascii():
        raise StoragePathError("GPU_ID must be an integer in [0, 999].")
    gpu_text = str(selected_gpu)
    if not gpu_text.isdecimal() or len(gpu_text) > 3:
        raise StoragePathError("GPU_ID must be an integer in [0, 999].")
    selected_gpu = int(gpu_text)

    package = _exportable_path(
        package_root if package_root is not None else source.get("PACKAGE_ROOT") or workspace_root()
    )
    runtime = _exportable_path(
        source.get("ASTRA_RUNTIME_ROOT") or package / ".runtime" / f"gpu_{selected_gpu}"
    )
    if any(character.isspace() for character in str(runtime)):
        raise StoragePathError(
            "ASTRA_RUNTIME_ROOT must not contain whitespace for Isaac Lab Kit arguments."
        )
    x11 = _exportable_path(source.get("ASTRA_X11_ROOT") or package / ".runtime" / "x11")
    cache = _exportable_path(source.get("ASTRA_CACHE_ROOT") or runtime / "cache")
    weights = _exportable_path(source.get("ASTRA_WEIGHTS_ROOT") or package / "weights")
    hf_home = weights / "huggingface"
    hf_datasets = cache / "huggingface" / "datasets"

    # Every value is chosen here, rather than trusting inherited cache overrides
    # that may point at another storage root or the user's home directory.
    path_settings = {
        "ASTRA_RUNTIME_ROOT": runtime,
        "ASTRA_X11_ROOT": x11,
        "ASTRA_CACHE_ROOT": cache,
        "ASTRA_WEIGHTS_ROOT": weights,
        "XDG_RUNTIME_DIR": runtime / "xdg",
        "XDG_CACHE_HOME": cache / "xdg",
        "XDG_CONFIG_HOME": runtime / "config",
        "XDG_DATA_HOME": runtime / "data",
        "XDG_STATE_HOME": runtime / "state",
        "TMPDIR": runtime / "tmp",
        "TMP": runtime / "tmp",
        "TEMP": runtime / "tmp",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
        "TORCH_HOME": weights / "torch",
        "TORCH_HUB": weights / "torch" / "hub",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "PYTORCH_KERNEL_CACHE_PATH": cache / "torch_kernels",
        "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
        "NUMBA_CACHE_DIR": cache / "numba",
        "WARP_CACHE_PATH": cache / "warp",
        "MPLCONFIGDIR": cache / "matplotlib",
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "HF_ASSETS_CACHE": cache / "huggingface" / "assets",
        "HUGGINGFACE_ASSETS_CACHE": cache / "huggingface" / "assets",
        "HF_DATASETS_CACHE": hf_datasets,
        "HF_MODULES_CACHE": cache / "huggingface" / "modules",
        "HF_XET_CACHE": cache / "huggingface" / "xet",
        "TRANSFORMERS_CACHE": hf_home / "hub",
        "HF_DATASETS_DOWNLOADED_DATASETS_PATH": hf_datasets / "downloads",
        "HF_DATASETS_EXTRACTED_DATASETS_PATH": hf_datasets / "downloads" / "extracted",
        "OMNI_CONFIG_PATH": runtime / "config" / "omniverse",
    }
    validated = {name: _exportable_path(path) for name, path in path_settings.items()}
    extra_directories = [
        package,
        _exportable_path(runtime / "kit"),
        _exportable_path(runtime / "data" / "omni"),
        _exportable_path(cache / "omni"),
        _exportable_path(runtime / "logs" / "omni"),
        _exportable_path(x11 / "packages"),
    ]
    directories = set(validated.values()) | set(extra_directories)
    for directory in directories:
        _check_directory(directory)

    omni_file = _exportable_path(validated["OMNI_CONFIG_PATH"] / "omniverse.toml")
    _check_file(omni_file)
    for name in ("package_info.txt", "package_sha256.txt"):
        _check_file(_exportable_path(x11 / name))
    # dpkg extraction can otherwise follow an existing usr/lib/share symlink
    # into another storage root even though the extraction root itself is inside configured workspace.
    _validate_existing_symlinks(x11)
    settings = {name: str(path) for name, path in validated.items()}
    settings["OMNICLIENT_HUB_MODE"] = "disabled"
    if dry_run:
        return settings

    # All path and file-type checks above happen before this first write.
    for directory in sorted(directories, key=lambda item: (len(item.parts), str(item))):
        directory.mkdir(parents=True, exist_ok=True)
    validated["XDG_RUNTIME_DIR"].chmod(0o700)
    content = "[paths]\n" + "".join(
        f"{name} = {json.dumps(str(path), ensure_ascii=False)}\n"
        for name, path in (
            ("data_root", extra_directories[2]),
            ("cache_root", extra_directories[3]),
            ("logs_root", extra_directories[4]),
        )
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=omni_file.parent,
            prefix=".omniverse.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, omni_file)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    """Validate paths or emit safe runtime settings for launcher consumption."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Check paths without creating them")
    validate.add_argument("--json", action="store_true", help="Emit a JSON array")
    validate.add_argument("paths", nargs="+")
    configure = subparsers.add_parser(
        "configure-runtime", help="Prepare configured workspace runtime storage"
    )
    configure.add_argument("--package-root", type=Path)
    configure.add_argument("--gpu-id")
    configure.add_argument("--dry-run", action="store_true")
    configure.add_argument("--json", action="store_true", help="Emit a JSON object")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            paths = [str(_exportable_path(path)) for path in args.paths]
            if args.json:
                print(json.dumps(paths, ensure_ascii=False))
            else:
                for path in paths:
                    print(path)
        else:
            settings = configure_runtime(
                args.package_root, gpu_id=args.gpu_id, dry_run=args.dry_run
            )
            if args.json:
                print(json.dumps(settings, ensure_ascii=False))
            else:
                for name, value in settings.items():
                    print(f"{name}={value}")
    except (StoragePathError, OSError) as exc:
        print(f"configured workspace storage configuration rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
