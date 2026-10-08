"""MkDocs hooks for the documentation site, registered in mkdocs.yml.

`on_pre_build` fails the build unless the API reference pages document each name in the package root's `__all__`
once, and nothing else, so the reference tracks the public API as names are added or removed.
"""
import ast
import re
from pathlib import Path
from typing import Any

# mkdocs comes from requirements-docs.txt, which the lint job does not install.
from mkdocs.exceptions import PluginError  # pylint: disable=import-error

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_INIT = REPO_ROOT / 'sd_backend_client' / '__init__.py'
REFERENCE_DIR = REPO_ROOT / 'docs' / 'reference'
DIRECTIVE_PATTERN = re.compile(r'^::: sd_backend_client\.(\w+)\s*$', re.MULTILINE)


def public_names() -> list[str]:
    """Return the package root's `__all__`, read from the source without importing the package."""
    tree = ast.parse(PACKAGE_INIT.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__'
                                                for target in node.targets):
            return list(ast.literal_eval(node.value))
    raise PluginError(f'No __all__ found in {PACKAGE_INIT}')


def documented_names() -> list[str]:
    """Return each name a `::: sd_backend_client.NAME` directive documents in the API reference pages."""
    names = []
    for page in sorted(REFERENCE_DIR.glob('*.md')):
        names.extend(DIRECTIVE_PATTERN.findall(page.read_text(encoding='utf-8')))
    return names


def on_pre_build(**_kwargs: Any) -> None:
    """Fail the build when the API reference and the root `__all__` disagree."""
    expected = public_names()
    documented = documented_names()
    problems = []
    missing = [name for name in expected if name not in documented]
    if missing:
        problems.append(f'not documented: {", ".join(missing)}')
    extra = sorted(set(documented) - set(expected))
    if extra:
        problems.append(f'documented but not in __all__: {", ".join(extra)}')
    repeated = sorted({name for name in documented if documented.count(name) > 1})
    if repeated:
        problems.append(f'documented more than once: {", ".join(repeated)}')
    if problems:
        raise PluginError(f'docs/reference/ does not match sd_backend_client.__all__ ({"; ".join(problems)}). Add a '
                          '"::: sd_backend_client.NAME" directive for each public name to a page under '
                          'docs/reference/.')
