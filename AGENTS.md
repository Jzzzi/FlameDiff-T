# Repository Guidelines

## Project Structure & Module Organization
This checkout is currently minimal and does not yet define a source tree. As code is added, keep runtime code in `src/`, tests in `tests/`, reusable assets in `assets/`, and longer design notes in `docs/`. Mirror code paths in tests when possible; for example, `src/foo/bar.py` should usually pair with `tests/foo/test_bar.py`.

## Build, Test, and Development Commands
No build or test automation is committed in the current workspace. If you introduce tooling, add a single stable entry point and document it in the repository root. Good patterns are `make test`, `make lint`, or direct Python commands such as `python -m pytest`. Keep commands deterministic so they can be reused in CI without local-only setup.

## Coding Style & Naming Conventions
Use 4-space indentation for Python and keep lines readable, ideally under 100 characters. Prefer `snake_case` for files, modules, and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep modules focused, avoid implicit side effects at import time, and add formatter or linter configuration together with the first code that depends on it.

## Testing Guidelines
Add tests for every behavior change and bug fix. Place unit tests under `tests/` and name files `test_<module>.py`. Prefer small, isolated tests over large integration fixtures unless the feature requires end-to-end coverage. When you add a new dependency or execution path, also add the command needed to run its tests locally.

## Commit & Pull Request Guidelines
This directory does not include local Git history, so no project-specific commit style can be inferred. Use short, imperative commit subjects such as `add dataset loader` or `fix tensor shape bug`. Pull requests should explain the change, note any setup or migration impact, link the related task if one exists, and include logs or screenshots when behavior is easiest to review visually.

## Repository Hygiene
Do not commit secrets, machine-specific paths, or large generated artifacts. Check in small reproducible fixtures instead of derived outputs, and document required environment variables in the repository root when they are introduced.
