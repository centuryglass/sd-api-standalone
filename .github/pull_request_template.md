<!-- Title: Conventional Commits, e.g. "fix(comfyui): keep the requested upscale size". Add "!" for a breaking change. -->

## Summary

<!-- What changes for a library user or contributor, and why. -->

Closes #

## Checklist

- [ ] Breaking change to the public API (the title is marked with `!`)
- [ ] Unit test added or updated in `tests/unit/` for any change to an emitted request or node graph
- [ ] `pytest tests/`, `python scripts/pylint_check.py` (Python 3.13) and `mypy` pass locally
- [ ] Integration-tested against: <!-- backend and version, e.g. ComfyUI v0.3.40; or "not needed: no HTTP client change" -->
