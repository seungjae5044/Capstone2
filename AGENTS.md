# Repository Guidelines

## Project Structure & Module Organization
- Python backend lives at `whisper_web_ui.py` (FastAPI + WebSocket). Supporting code under `model/`, configuration in `config/`, documentation in `doc/`, and artifacts in `output/`.
- Frontend lives in `src_F/` (Vite + React + TypeScript).
- Tests and runnable scripts are in `test/` (e.g., `test/test_whisper2.py`).
- Python project metadata is in `pyproject.toml`; Node metadata in `src_F/package.json`.

## Build, Test, and Development Commands
- Python install: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- Run API (dev): `uvicorn whisper_web_ui:app --reload` (serves http://localhost:8000).
- Manual audio test: `python test/test_whisper2.py --config config/config_whisper.json`.
- Frontend dev: `cd src_F && npm install && npm run dev` (http://localhost:5173). Build: `npm run build`.

## Coding Style & Naming Conventions
- Python: format with Black (line length 88). Use type hints, snake_case for functions/variables, PascalCase for classes, NumPy-style docstrings. Group imports: stdlib → third‑party → local, one per line. Prefer explicit exceptions and `logging` over prints.
- TypeScript/React: functional components with hooks; components/interfaces in PascalCase; group imports (React, third‑party, local). Use Tailwind classes consistently; avoid prop drilling—prefer composition or context.

## Testing Guidelines
- Python: prefer pytest with files in `test/` named `test_*.py`. Run all with `pytest -q` (when configured). The realtime script `test/test_whisper2.py` is for manual validation.
- Frontend: no tests configured; add lightweight component tests when introducing complex logic or utility functions.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat(api): add websocket metrics`, `fix(ui): debounce waveform redraw`). Keep messages imperative and scoped.
- PRs: include a clear summary, linked issues, verification steps/commands, and screenshots or clips for UI changes. Keep diffs focused; note any breaking changes.

## Security & Configuration Tips
- Do not commit credentials, API keys, model weights, or large artifacts; use `.env`/local paths and keep `output/` clean or gitignored.
- Runtime config: `config/config_whisper.json` (e.g., `force_device: cpu|cuda|mps`, sample rates, chunking). Document changes in PRs.
