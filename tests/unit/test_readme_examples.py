"""The README's Python examples and the scripts in `examples/` run against every recorded server, offline.

Each example runs with `connect_to_backend` patched to return a real client whose `get` replays a recording (see
`recorded_replay`), so discovery calls parse real responses. The `submit_*` methods are replaced by autospecced
stand-ins that check the call signature and return an already-finished handle. A renamed export, method or parameter
fails here.
"""
import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional
from unittest.mock import create_autospec

import pytest
from PIL import Image

import sd_backend_client
from sd_backend_client import A1111Webservice, Backend, ComfyUiWebservice, GenerationHandle, GenerationProgress, \
    GenerationResult, GenerationStatus, PreprocessorParams

from .recorded_replay import recordings, replay

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / 'README.md'
EXAMPLE_SCRIPTS = sorted((REPO_ROOT / 'examples').glob('*.py'))
SUBMIT_METHODS = ('submit_txt2img', 'submit_img2img', 'submit_inpaint', 'submit_upscale',
                  'submit_preprocessor_preview')
RECORDINGS = recordings('comfyui') + recordings('webui')


def _python_blocks() -> list[str]:
    """Every fenced ```python block in the README, in order."""
    return re.findall(r'^```python\n(.*?)^```', README.read_text(encoding='utf-8'), re.MULTILINE | re.DOTALL)


README_BLOCKS = _python_blocks()


class _FinishedHandle(GenerationHandle):
    """A handle whose job has already finished with one small image."""

    def __init__(self) -> None:
        super().__init__('finished-task')
        self.cancelled = False

    def poll(self) -> GenerationProgress:
        return GenerationProgress(GenerationStatus.FINISHED, progress=1.0)

    def _build_result(self) -> GenerationResult:
        return GenerationResult(images=[Image.new('RGB', (8, 8))], seeds=[1], seed=1, task_id=self.task_id)

    def cancel(self) -> bool:
        self.cancelled = True
        return True


def _offline_backend(recording: dict[str, Any]) -> Backend:
    """A real client for the recording's backend that replays its responses and finishes every job at once."""
    service: A1111Webservice | ComfyUiWebservice
    if recording['meta']['backend'] == 'comfyui':
        service = ComfyUiWebservice('http://127.0.0.1:8188', live_progress=False)
    else:
        service = A1111Webservice('http://127.0.0.1:7860')
    replay(service, recording)
    for name in SUBMIT_METHODS:
        stand_in = create_autospec(getattr(service, name), side_effect=lambda *_args, **_kwargs: _FinishedHandle())
        setattr(service, name, stand_in)
    return service


@pytest.fixture(name='connect_offline')
def fixture_connect_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run in an empty directory, and return a function that points `connect_to_backend` at a recording."""
    monkeypatch.chdir(tmp_path)

    def connect(recording: dict[str, Any]) -> Backend:
        backend = _offline_backend(recording)
        connect_stand_in = create_autospec(sd_backend_client.connect_to_backend, return_value=backend)
        monkeypatch.setattr(sd_backend_client, 'connect_to_backend', connect_stand_in)
        return backend

    return connect


def _create_opened_images(source: str, directory: Path) -> None:
    """Write a small placeholder for each file the code passes to `Image.open`."""
    for name in re.findall(r"Image\.open\('([^']+)'\)", source):
        Image.new('RGB', (512, 512), 'gray').save(directory / name)


def _load_example(path: Path) -> ModuleType:
    """Import an example script as a module, without running its `__main__` block."""
    spec = importlib.util.spec_from_file_location(f'example_{path.stem}', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_imports(source: str) -> list[tuple[str, list[str]]]:
    """The (module, imported names) of every import of the package or one of its modules in `source`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'sd_backend_client':
            found.append((node.module or '', [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names if alias.name.split('.')[0] == 'sd_backend_client']
    return found


def test_readme_has_python_examples():
    """The block extraction still finds the README's examples, so the tests below never silently run none."""
    assert len(README_BLOCKS) >= 5


@pytest.mark.parametrize('path', [README, *EXAMPLE_SCRIPTS], ids=lambda path: path.name)
def test_examples_import_only_the_public_api(path: Path):
    """Examples import from the package root, and only names in `__all__`, so they show the supported API."""
    source = path.read_text(encoding='utf-8')
    sources = README_BLOCKS if path == README else [source]
    imports = [found for code in sources for found in _package_imports(code)]
    assert imports, f'{path.name} imports nothing from sd_backend_client'
    for module, names in imports:
        assert module == 'sd_backend_client', f'{path.name} imports from internal module {module}'
        for name in names:
            assert name in sd_backend_client.__all__, f'{path.name} imports {name}, which is not in __all__'


@pytest.mark.parametrize('recording', RECORDINGS)
@pytest.mark.parametrize('block_index', range(len(README_BLOCKS)), ids=lambda index: f'block{index}')
def test_readme_example_runs(block_index: int, recording: dict[str, Any], connect_offline, tmp_path: Path,
                             capsys: pytest.CaptureFixture[str]):
    """Each README example runs to completion against each recorded server."""
    del capsys  # Silences the examples' prints.
    source = README_BLOCKS[block_index]
    backend = connect_offline(recording)
    _create_opened_images(source, tmp_path)
    exec(compile(source, f'README.md block {block_index}', 'exec'), {'__name__': '__readme__'})  # pylint: disable=exec-used
    if 'submit_' in source and 'connect_to_backend' in source:
        assert any(getattr(backend, name).called for name in SUBMIT_METHODS), 'the example submitted no job'


@pytest.mark.parametrize('recording', RECORDINGS)
def test_canny_example_sets_thresholds(recording: dict[str, Any], connect_offline, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """The canny preview script finds the server's canny preprocessor and passes it the requested thresholds."""
    del capsys
    backend = connect_offline(recording)
    _create_opened_images("Image.open('in.png')", tmp_path)
    example = _load_example(REPO_ROOT / 'examples' / 'canny_preprocessor_preview.py')
    monkeypatch.setattr(sys, 'argv', ['canny', 'in.png', 'out.png', '--low', '42', '--high', '99'])

    example.main()

    assert (tmp_path / 'out.png').exists()
    submit: Any = backend.submit_preprocessor_preview
    params: Optional[PreprocessorParams] = submit.call_args.args[1]
    assert isinstance(params, PreprocessorParams)
    assert sorted(value for value in params.parameter_values.values() if value in (42, 99)) == [42, 99]
