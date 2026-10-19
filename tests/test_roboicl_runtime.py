"""Exercise codec initialization when Kit exposes a second Pillow installation."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_keeps_jpeg_plugin_with_original_pillow_core():
    kit_dirs = list(
        (
            Path(sys.prefix)
            / "lib"
            / ("python" + ".".join(map(str, sys.version_info[:2])))
            / "site-packages/isaacsim/extscache"
        ).glob("omni.kit.pip_archive-*/pip_prebundle/PIL")
    )
    if not kit_dirs:
        pytest.skip("Requires the shared Isaac Sim runtime")
    from tactalent.runtime import BOOTSTRAP

    bootstrap = BOOTSTRAP.format(module="tactalent.roboicl.run")
    program = """
import json, runpy, sys
from pathlib import Path
from io import BytesIO
import PIL
from PIL import Image
def check_entrypoint(name, run_name):
    assert name == "tactalent.roboicl.run" and run_name == "__main__"
    PIL.__path__.insert(0, sys.argv[1])
    image = Image.new("RGB", (640, 510), (12, 34, 56))
    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=90, subsampling=0)
    encoded.seek(0)
    decoded = Image.open(encoded)
    decoded.load()
    assert decoded.size == image.size
    from PIL import JpegImagePlugin, ImageFile
    original = Path(Image.__file__).parent
    assert Path(JpegImagePlugin.__file__).parent == original
    assert Path(ImageFile.__file__).parent == original
    print(json.dumps({"encoded_bytes": len(encoded.getvalue())}))
runpy.run_module = check_entrypoint
exec(sys.argv[2])
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(kit_dirs[0]), bootstrap],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["encoded_bytes"] > 0
