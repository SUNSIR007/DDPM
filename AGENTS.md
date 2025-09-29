# Repository Guidelines

## Project Structure & Module Organization
- Core training code lives in `ddpm_conditional.py`, with model components in `modules.py` and shared utilities in `utils.py`.
- Monitoring helpers (`early_stopping_monitor.py`, `mps_training_monitor.py`) and resume scripts guard long runs; keep generated checkpoints in `models/DDPM_conditional/`.
- Data assets belong under `datasets/`; generated samples and TensorBoard logs should stay in `results/DDPM_conditional/` and `runs/DDPM_conditional/`.
- Demo notebooks or experiments belong in `demo_results/` or a new subfolder—avoid polluting the repository root.

## Build, Test, and Development Commands
- `python -m venv ddpm_env && source ddpm_env/bin/activate` to enter the managed environment; install deps with `pip install -r requirements.txt`.
- `python ddpm_conditional.py --help` inspects all training switches; run `python ddpm_conditional.py` for default CIFAR-10 conditional training.
- `bash run_ddpm.sh` launches the preconfigured CUDA workflow; use `python run_ddpm_mps.py` for Apple Silicon setups.
- `python resume_training.py --run-name DDPM_conditional` resumes from interrupted checkpoints inside `models/DDPM_conditional/`.

## Coding Style & Naming Conventions
- Follow PEP 8 defaults: 4-space indentation, snake_case for functions, PascalCase for classes, and keep constants uppercase.
- Match the bilingual docstring style already in `modules.py`; explain non-obvious math in concise comments rather than inline narration.
- Prefer descriptive argument names (`noise_steps`, `cfg_scale`) and reuse helper utilities from `utils.py` instead of reimplementing I/O or logging.

## Testing Guidelines
- Run `python test_mps_setup.py` after dependency or device changes to verify Torch/MPS availability; it should complete without stack traces.
- Use `python simple_test_early_stopping.py` (quick smoke) or `python test_early_stopping.py` (full sweep, writes plots to `test_early_stopping_results/`) when touching monitoring logic.
- Place new tests as executable Python scripts prefixed with `test_` at the repository root, mirroring the existing convention; print salient metrics or asset paths for reviewers.

## Commit & Pull Request Guidelines
- Keep commits small, focused, and present-tense (e.g., `Add EMA cold-start guard`); emoji prefixes are optional but keep the English summary clear.
- Reference related issues in the body, describe training metrics, and attach sample grids saved under `demo_results/` or `results/` when behavior changes.
- Pull requests should state device/command used, list regression tests executed, and flag any follow-up work; include screenshots or image paths for generated outputs.

## Environment & Configuration Tips
- Record any local path overrides (e.g., custom dataset roots) in `.env` or document them in the PR instead of hard-coding absolute paths.
- Before long runs, verify storage targets exist (`models/`, `results/`, `runs/`) to avoid silent failures, and prefer incremental checkpoints over overwriting baselines.
