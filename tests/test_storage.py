"""Storage boundaries checked with real paths and symbolic links."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from tactalent.utils import storage


@contextlib.contextmanager
def working_directory(path):
    """Temporarily change cwd, including under the server's Python 3.10."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.mount = self.root / "workspace"
        self.mount.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.mount_patch = patch.object(storage, "OUTPUT_ROOT", self.mount)
        self.mount_patch.start()

    def tearDown(self):
        self.mount_patch.stop()
        self.temp.cleanup()

    def test_missing_output_path_is_validated_without_creation(self):
        target = self.mount / "project" / "results" / "run" / "result.json"
        self.assertEqual(storage.require_output_path(str(target)), target)
        self.assertFalse((self.mount / "project").exists())
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_existing_output_and_mount_root_are_accepted(self):
        target = self.mount / "result.json"
        target.write_text("existing")
        self.assertEqual(storage.require_output_path(target), target)
        self.assertEqual(storage.require_output_path(self.mount), self.mount)
        self.assertEqual(target.read_text(), "existing")

    def test_relative_path_from_workspace_is_canonicalized_without_creation(self):
        with working_directory(self.mount):
            result = storage.require_output_path(Path("new") / ".." / "results")
        self.assertEqual(result, self.mount / "results")
        self.assertFalse(result.exists())

    def test_relative_path_from_outside_is_rejected(self):
        with working_directory(self.outside):
            with self.assertRaises(storage.StoragePathError):
                storage.require_output_path("results")
        self.assertFalse((self.outside / "results").exists())

    def test_outside_home_and_mount_prefix_collision_are_rejected(self):
        paths = [
            self.outside / "results",
            self.root / "home" / "user" / "results",
            self.root / "workspace0" / "results",
        ]
        for target in paths:
            with self.subTest(path=target):
                with self.assertRaises(storage.StoragePathError):
                    storage.require_output_path(target)
                self.assertFalse(target.exists())

    def test_parent_traversal_outside_is_rejected(self):
        with self.assertRaises(storage.StoragePathError):
            storage.require_output_path(self.mount / ".." / "outside" / "results")

    def test_symlink_to_outside_existing_directory_is_rejected(self):
        link = self.mount / "cache"
        link.symlink_to(self.outside, target_is_directory=True)
        with self.assertRaises(storage.StoragePathError):
            storage.require_output_path(link / "torch" / "new.bin")
        self.assertFalse((self.outside / "torch").exists())

    def test_symlink_to_missing_outside_directory_is_rejected(self):
        link = self.mount / "cache"
        target = self.outside / "not_created"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(storage.StoragePathError):
            storage.require_output_path(link / "new.bin")
        self.assertFalse(target.exists())

    def test_symlink_within_mount_returns_real_destination(self):
        real = self.mount / "real"
        real.mkdir()
        link = self.mount / "alias"
        link.symlink_to(real, target_is_directory=True)
        self.assertEqual(storage.require_output_path(link / "result.json"), real / "result.json")
        self.assertFalse((real / "result.json").exists())

    def test_symlink_loop_is_a_storage_error(self):
        link = self.mount / "loop"
        link.symlink_to(link)
        with self.assertRaises(storage.StoragePathError):
            storage.require_output_path(link / "results")

    def test_mount_name_symlink_does_not_allow_another_disk(self):
        linked_mount = self.root / "linked_workspace"
        linked_mount.symlink_to(self.outside, target_is_directory=True)
        with patch.object(storage, "OUTPUT_ROOT", linked_mount):
            with self.assertRaises(storage.StoragePathError):
                storage.require_output_path(linked_mount / "results")

    def test_invalid_path_inputs_are_rejected(self):
        for target in ("", None, 7, [], b"/opt/tactalent/storage/results", "bad\x00path"):
            with self.subTest(path=target):
                with self.assertRaises(storage.StoragePathError):
                    storage.require_output_path(target)

    def test_user_home_expansion_outside_is_rejected(self):
        with self.assertRaises(storage.StoragePathError):
            storage.require_output_path("~/results")

    def test_runtime_default_environment_and_omniverse_paths(self):
        package = self.mount / "project"
        settings = storage.configure_runtime(package, gpu_id="02", environment={})
        runtime = package / ".runtime" / "gpu_2"
        self.assertEqual(settings["ASTRA_RUNTIME_ROOT"], str(runtime))
        self.assertEqual(settings["ASTRA_CACHE_ROOT"], str(runtime / "cache"))
        self.assertEqual(settings["ASTRA_WEIGHTS_ROOT"], str(package / "weights"))
        self.assertEqual(settings["HF_HOME"], str(package / "weights" / "huggingface"))
        self.assertEqual(settings["OMNICLIENT_HUB_MODE"], "disabled")
        for name, value in settings.items():
            if name != "OMNICLIENT_HUB_MODE":
                self.assertTrue(Path(value).is_dir(), name)
                self.assertEqual(storage.require_output_path(value), Path(value))
        self.assertEqual((runtime / "xdg").stat().st_mode & 0o777, 0o700)
        content = (Path(settings["OMNI_CONFIG_PATH"]) / "omniverse.toml").read_text()
        self.assertEqual(
            content,
            "[paths]\n"
            + f'data_root = "{runtime / "data" / "omni"}"\n'
            + f'cache_root = "{runtime / "cache" / "omni"}"\n'
            + f'logs_root = "{runtime / "logs" / "omni"}"\n',
        )
        self.assertFalse(list(Path(settings["OMNI_CONFIG_PATH"]).glob("*.tmp")))

    def test_runtime_overrides_and_legacy_cache_values_are_replaced(self):
        package = self.mount / "project"
        overrides = {
            "ASTRA_RUNTIME_ROOT": str(self.mount / "runtime"),
            "ASTRA_X11_ROOT": str(self.mount / "x11"),
            "ASTRA_CACHE_ROOT": str(self.mount / "cache"),
            "ASTRA_WEIGHTS_ROOT": str(self.mount / "weights"),
        }
        inherited = {
            name: str(self.outside / name)
            for name in (
                "TMPDIR",
                "TORCH_EXTENSIONS_DIR",
                "PYTORCH_KERNEL_CACHE_PATH",
                "NUMBA_CACHE_DIR",
                "WARP_CACHE_PATH",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HUGGINGFACE_HUB_CACHE",
                "TRANSFORMERS_CACHE",
                "HF_MODULES_CACHE",
                "HF_DATASETS_DOWNLOADED_DATASETS_PATH",
                "HF_DATASETS_EXTRACTED_DATASETS_PATH",
                "OMNI_CONFIG_PATH",
            )
        }
        environment = {**overrides, **inherited}
        before = dict(environment)
        settings = storage.configure_runtime(package, environment=environment)
        for name, value in overrides.items():
            self.assertEqual(settings[name], value)
        for name, value in inherited.items():
            self.assertNotEqual(settings[name], value)
            storage.require_output_path(settings[name])
            self.assertFalse(Path(value).exists())
        self.assertEqual(environment, before)

    def test_credentials_are_neither_read_nor_returned(self):
        class CredentialGuard(Mapping):
            def __getitem__(self, key):
                if key in {
                    "OPENROUTER_API_KEY",
                    "ASTRA_API_KEY",
                    "HF_TOKEN_PATH",
                    "HF_TOKEN",
                }:
                    raise AssertionError(f"credential was accessed: {key}")
                raise KeyError(key)

            def __iter__(self):
                raise AssertionError("environment was enumerated")

            def __len__(self):
                raise AssertionError("environment was enumerated")

        settings = storage.configure_runtime(
            self.mount / "project", environment=CredentialGuard(), dry_run=True
        )
        self.assertFalse(any("TOKEN" in name or "API_KEY" in name for name in settings))

    def test_runtime_dry_run_does_not_create_directories_or_configuration(self):
        package = self.mount / "project"
        settings = storage.configure_runtime(package, environment={}, dry_run=True)
        self.assertIn("WARP_CACHE_PATH", settings)
        self.assertFalse(package.exists())
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_outside_root_override_is_rejected_before_any_creation(self):
        package = self.mount / "project"
        for name in (
            "ASTRA_RUNTIME_ROOT",
            "ASTRA_X11_ROOT",
            "ASTRA_CACHE_ROOT",
            "ASTRA_WEIGHTS_ROOT",
        ):
            with self.subTest(name=name):
                with self.assertRaises(storage.StoragePathError):
                    storage.configure_runtime(
                        package, environment={name: str(self.outside / "new")}
                    )
                self.assertFalse(package.exists())
                self.assertFalse((self.outside / "new").exists())

    def test_derived_symlink_escape_is_rejected_before_any_creation(self):
        package = self.mount / "project"
        cache = self.mount / "cache"
        cache.mkdir()
        (cache / "torch_extensions").symlink_to(self.outside, target_is_directory=True)
        before = set(self.root.rglob("*"))
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(package, environment={"ASTRA_CACHE_ROOT": str(cache)})
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_omniverse_file_symlink_escape_is_rejected_before_any_creation(self):
        package = self.mount / "project"
        omni = package / ".runtime" / "gpu_0" / "config" / "omniverse"
        omni.mkdir(parents=True)
        (omni / "omniverse.toml").symlink_to(self.outside / "new.toml")
        before = set(self.root.rglob("*"))
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(package, environment={})
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_x11_packages_symlink_escape_rejected_before_runtime_creation(self):
        package = self.mount / "project"
        x11 = self.mount / "x11"
        x11.mkdir()
        (x11 / "packages").symlink_to(self.outside, target_is_directory=True)
        before = set(self.root.rglob("*"))
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(package, environment={"ASTRA_X11_ROOT": str(x11)})
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertFalse(package.exists())

    def test_x11_metadata_symlink_escape_rejected_before_runtime_creation(self):
        package = self.mount / "project"
        x11 = self.mount / "x11"
        x11.mkdir()
        for name in ("package_info.txt", "package_sha256.txt"):
            with self.subTest(name=name):
                link = x11 / name
                link.symlink_to(self.outside / name)
                before = set(self.root.rglob("*"))
                with self.assertRaises(storage.StoragePathError):
                    storage.configure_runtime(package, environment={"ASTRA_X11_ROOT": str(x11)})
                self.assertEqual(set(self.root.rglob("*")), before)
                self.assertFalse(package.exists())
                link.unlink()

    def test_x11_usr_symlink_escape_rejected_before_runtime_creation(self):
        package = self.mount / "project"
        x11 = self.mount / "x11"
        x11.mkdir()
        (x11 / "usr").symlink_to(self.outside, target_is_directory=True)
        before = set(self.root.rglob("*"))
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(package, environment={"ASTRA_X11_ROOT": str(x11)})
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertFalse(package.exists())

    def test_x11_nested_link_target_escape_is_checked(self):
        package = self.mount / "project"
        x11 = self.mount / "x11"
        x11.mkdir()
        real_usr = self.mount / "usr"
        real_usr.mkdir()
        (real_usr / "lib").symlink_to(self.outside, target_is_directory=True)
        (x11 / "usr").symlink_to(real_usr, target_is_directory=True)
        before = set(self.root.rglob("*"))
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(package, environment={"ASTRA_X11_ROOT": str(x11)})
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_x11_wrong_packages_and_metadata_types_rejected_before_writes(self):
        package = self.mount / "project"
        x11 = self.mount / "x11"
        x11.mkdir()
        packages = x11 / "packages"
        packages.write_text("keep")
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(package, environment={"ASTRA_X11_ROOT": str(x11)})
        self.assertFalse(package.exists())
        packages.unlink()
        for name in ("package_info.txt", "package_sha256.txt"):
            with self.subTest(name=name):
                wrong_type = x11 / name
                wrong_type.mkdir()
                with self.assertRaises(storage.StoragePathError):
                    storage.configure_runtime(package, environment={"ASTRA_X11_ROOT": str(x11)})
                self.assertFalse(package.exists())
                self.assertFalse(packages.exists())
                wrong_type.rmdir()

    def test_x11_workspace_symlinks_and_directory_cycle_are_safe(self):
        package = self.mount / "project"
        x11 = self.mount / "x11"
        x11.mkdir()
        real_usr = self.mount / "usr"
        real_usr.mkdir()
        (real_usr / "share").mkdir()
        (real_usr / "share" / "back").symlink_to(x11, target_is_directory=True)
        (x11 / "usr").symlink_to(real_usr, target_is_directory=True)
        (x11 / "package_info.txt").write_text("existing metadata")
        settings = storage.configure_runtime(package, environment={"ASTRA_X11_ROOT": str(x11)})
        self.assertEqual(settings["ASTRA_X11_ROOT"], str(x11))
        self.assertTrue((x11 / "packages").is_dir())
        self.assertEqual((x11 / "package_info.txt").read_text(), "existing metadata")

    def test_file_in_directory_path_is_rejected_before_any_creation(self):
        package = self.mount / "project"
        cache_file = self.mount / "cache_file"
        cache_file.write_text("unchanged")
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(
                package, environment={"ASTRA_CACHE_ROOT": str(cache_file / "cache")}
            )
        self.assertFalse(package.exists())
        self.assertEqual(cache_file.read_text(), "unchanged")

    def test_line_break_path_and_invalid_gpu_are_rejected_without_writes(self):
        package = self.mount / "project"
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(
                package,
                environment={"ASTRA_CACHE_ROOT": str(self.mount / "bad\ncache")},
            )
        for gpu in (True, "-1", "1000", "1.0", "١"):
            with self.subTest(gpu=gpu):
                with self.assertRaises(storage.StoragePathError):
                    storage.configure_runtime(package, gpu_id=gpu, environment={})
        self.assertFalse(package.exists())

    def test_runtime_whitespace_is_rejected_before_any_creation(self):
        package = self.mount / "project"
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(
                package,
                environment={"ASTRA_RUNTIME_ROOT": str(self.mount / "has space")},
            )
        self.assertFalse(package.exists())
        self.assertFalse((self.mount / "has space").exists())

    def test_runtime_symlink_to_whitespace_directory_is_rejected(self):
        package = self.mount / "project"
        real = self.mount / "has space"
        real.mkdir()
        alias = self.mount / "runtime"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(storage.StoragePathError):
            storage.configure_runtime(package, environment={"ASTRA_RUNTIME_ROOT": str(alias)})
        self.assertFalse(package.exists())
        self.assertEqual(list(real.iterdir()), [])

    def test_omniverse_toml_quotes_are_escaped(self):
        package = self.mount / 'project"name'
        settings = storage.configure_runtime(package, environment={})
        content = (Path(settings["OMNI_CONFIG_PATH"]) / "omniverse.toml").read_text()
        self.assertIn('project\\"name', content)
        self.assertNotIn('project"name', content)

    def test_cli_validate_outputs_canonical_paths_and_json_without_creation(self):
        targets = [self.mount / "one", self.mount / "two"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = storage.main(["validate", *(str(path) for path in targets)])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().splitlines(), [str(path) for path in targets])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = storage.main(["validate", "--json", *(str(path) for path in targets)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue()), [str(path) for path in targets])
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_cli_validate_bad_second_path_does_not_emit_partial_output(self):
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            status = storage.main(["validate", str(self.mount / "good"), str(self.outside)])
        self.assertEqual(status, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("rejected", error.getvalue())

    def test_cli_configure_dry_run_emits_safe_assignment_lines(self):
        package = self.mount / "project"
        with (
            patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = storage.main(
                [
                    "configure-runtime",
                    "--package-root",
                    str(package),
                    "--gpu-id",
                    "4",
                    "--dry-run",
                ]
            )
        self.assertEqual(status, 0)
        settings = dict(line.split("=", 1) for line in output.getvalue().splitlines())
        self.assertEqual(settings["ASTRA_RUNTIME_ROOT"], str(package / ".runtime" / "gpu_4"))
        self.assertEqual(settings["OMNICLIENT_HUB_MODE"], "disabled")
        self.assertFalse(package.exists())

    def test_cli_configure_json_creates_only_workspace_directories(self):
        package = self.mount / "project"
        with (
            patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = storage.main(["configure-runtime", "--package-root", str(package), "--json"])
        self.assertEqual(status, 0)
        settings = json.loads(output.getvalue())
        self.assertTrue((Path(settings["OMNI_CONFIG_PATH"]) / "omniverse.toml").is_file())
        self.assertEqual(list(self.outside.iterdir()), [])

    def test_storage_cli_import_is_dependency_free(self):
        script = """
import importlib.abc
import sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'PIL', 'isaaclab', 'isaacsim', 'curobo'}:
            raise AssertionError('storage attempted heavy import: ' + fullname)
sys.meta_path.insert(0, RejectHeavy())
from tactalent.utils.storage import main
raise SystemExit(main(['--help']))
"""
        result = subprocess.run(
            [sys.executable, "-S", "-B", "-c", script],
            cwd=Path(storage.__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("configure-runtime", result.stdout)


if __name__ == "__main__":
    unittest.main()
