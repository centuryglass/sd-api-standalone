"""Importing the package must not create files or directories outside the process."""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_creates_no_directories(tmp_path: Path) -> None:
    """Importing intrapaint_api leaves an empty HOME and XDG data/state dirs untouched."""
    env = {**os.environ, 'HOME': str(tmp_path), 'USERPROFILE': str(tmp_path),
           'XDG_DATA_HOME': str(tmp_path / 'data'), 'XDG_STATE_HOME': str(tmp_path / 'state'),
           'XDG_CACHE_HOME': str(tmp_path / 'cache'), 'XDG_CONFIG_HOME': str(tmp_path / 'config'),
           'PYTHONPATH': str(REPO_ROOT), 'PYTHONDONTWRITEBYTECODE': '1'}
    code = ('import intrapaint_api, intrapaint_api.util.shared_constants, '
            'intrapaint_api.api.a1111_webservice, intrapaint_api.api.comfyui_webservice')
    result = subprocess.run([sys.executable, '-c', code], env=env, cwd=tmp_path, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
