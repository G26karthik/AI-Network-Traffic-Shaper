# Contributing

Thanks for your interest in contributing! This project targets Windows and PowerShell usage.

## Getting started
- Fork the repository and create a feature branch from `main`.
- Use a Python virtual environment (see README installation steps).
- Ensure Wireshark/TShark is installed and available in PATH.
- Run scripts from an elevated PowerShell when capturing or shaping.

## Coding guidelines
- Keep Windows support first-class (PowerShell examples, `tshark` availability checks).
- Prefer small, single-purpose scripts with clear argparse help.
- If you change CLIs or behavior, update README and docs/ accordingly.
- Avoid breaking changes to filenames/entry points without discussion.

## Testing
- Manual smoke tests are acceptable for now:
  - `capture_features.py --list` works and capture terminates with `--duration`.
  - `traffic_generator.py` runs in `--method socket` on loopback.
  - `train_model.py` trains on a small `dataset.csv` and saves `traffic_model.pkl`.
  - `predict_and_shape.py` loads the model and prints predictions (without `--shape`).
- Please include sample outputs or logs in PRs.

## Pull Requests
- Link to an issue or describe the motivation and context.
- Keep changes focused; separate unrelated fixes into different PRs.
- The maintainers may request adjustments for clarity or Windows compatibility.
