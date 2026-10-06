"""CI packaging gate: fail when the built sdist or wheel is missing a subpackage or the py.typed marker.

setuptools only packages what `[tool.setuptools.packages.find]` and `[tool.setuptools.package-data]` in pyproject.toml
select, so a new subpackage or data file can be left out of the distribution while every test still passes against
the checkout. This compares the archives in a dist directory against the checkout's source tree:
- Every directory under sd_backend_client/ holding an __init__.py must appear, with its __init__.py, in both the
  wheel and the sdist.
- sd_backend_client/py.typed must appear in both, so type checkers read the installed package's annotations.

Usage: python scripts/check_dist.py [DIST_DIR]   (default: dist)
"""
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PACKAGE = 'sd_backend_client'


def _required_files() -> set[str]:
    """Return the package files every distribution must contain, as paths relative to the project directory."""
    required = {f'{PACKAGE}/py.typed'}
    for init_file in (PROJECT_DIR / PACKAGE).rglob('__init__.py'):
        required.add(init_file.relative_to(PROJECT_DIR).as_posix())
    return required


def _single(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        sys.exit(f'Expected one {pattern} in {dist_dir}, found {len(matches)}: {[m.name for m in matches]}')
    return matches[0]


def _wheel_files(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _sdist_files(sdist: Path) -> set[str]:
    """Return the sdist's member paths with the top-level `name-version/` directory stripped."""
    with tarfile.open(sdist) as archive:
        return {name.split('/', 1)[1] for name in archive.getnames() if '/' in name}


def main() -> int:
    """Check the wheel and sdist in the given dist directory, printing what each is missing."""
    dist_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_DIR / 'dist'
    required = _required_files()
    failed = False
    wheel = _single(dist_dir, '*.whl')
    sdist = _single(dist_dir, '*.tar.gz')
    for archive, members in ((wheel, _wheel_files(wheel)), (sdist, _sdist_files(sdist))):
        missing = sorted(required - members)
        if missing:
            failed = True
            print(f'{archive.name} is missing:')
            for path in missing:
                print(f'  {path}')
        else:
            print(f'{archive.name}: all {len(required)} required files present')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
