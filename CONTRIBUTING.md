# Contributing to sd-backend-client

Bug reports, fixes and improvements are welcome. Open work is tracked in
[GitHub issues](https://github.com/centuryglass/sd-backend-client/issues); check there before starting on something
larger, and open an issue first for a change to the public API.

Coding agents follow [`AGENTS.md`](AGENTS.md), which holds the full set of repository rules. This file is the short
version for people.

## Setup

Python 3.11 or newer is required.

```bash
git clone https://github.com/centuryglass/sd-backend-client.git
cd sd-backend-client
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -e .
```

`requirements-dev.txt` pins the test, lint and type-check tools CI uses. Runtime dependencies are listed in both
`pyproject.toml` and `requirements.txt`; keep the two lists in sync, and ask in an issue before adding one, since every
runtime dependency lands in each library user's environment.

## Checks

CI runs these on every pull request, and its `ci` job must pass before a PR merges. Run them locally first:

```bash
pytest tests/                    # unit tests, plus integration tests for any backend that is running
python scripts/pylint_check.py   # lint: fails only on messages beyond scripts/pylint_baseline.json
mypy                             # type check, configured in pyproject.toml's [tool.mypy]
```

- **Lint on Python 3.13.** The pylint baseline is only valid for the Python version it was generated with. After
  fixing existing messages, run `python scripts/pylint_check.py --update-baseline` and commit the updated baseline.
- **Code must run on Python 3.11.** Avoid 3.12-only syntax such as the `type X = ...` statement.

## Tests

- **Unit tests (`tests/unit/`)** are offline and deterministic: no server, no GPU, no network. They cover the code
  that turns parameters into request bodies and ComfyUI node graphs. A change to a builder or to request
  serialization comes with a unit test that would fail if the emitted request changed.
- **Integration tests** (the rest of `tests/`) talk to a real ComfyUI or A1111/Forge server and skip themselves when
  none is reachable, so CI never runs them. A change to the HTTP clients needs a local run against a live backend:

  ```bash
  SD_API_URL=http://127.0.0.1:7860 COMFYUI_API_URL=http://127.0.0.1:8188 pytest tests/
  pytest tests/ --run-generation   # adds slow real-diffusion round-trips; needs a loaded checkpoint
  ```

  [`tests/README.md`](tests/README.md) lists every environment variable and what each test file covers. Say in the
  PR which backend and version you tested against.

## Pull requests

- Branch from `main` and target your PR at `main`.
- **Titles use [Conventional Commits](https://www.conventionalcommits.org/):** `type: summary`, for example
  `fix(comfyui): keep the requested upscale size`. Common types are `feat`, `fix`, `docs`, `refactor`, `perf`,
  `test`, `build`, `ci` and `chore`. Mark a change that breaks the public API with `!` (`feat!: ...`).
- PRs are squash-merged, and [release-please](https://github.com/googleapis/release-please) turns the title into the
  changelog entry and version bump. Describe the change from a library user's point of view, and don't edit the
  version or `CHANGELOG.md` by hand.
- A PR that closes an issue says `Closes #NN` in its description.

## Reporting security issues

Don't open a public issue for a vulnerability. [`SECURITY.md`](SECURITY.md) explains how to report one privately.
