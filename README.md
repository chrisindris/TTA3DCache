# TTA3DCache

Basic starter template for a deep learning paper codebase built around **PyTorch**.

## Project layout

```text
TTA3DCache/
├── configs/                 # Experiment configs
├── data/                    # Local datasets (ignored or linked)
├── notebooks/               # Analysis notebooks
├── scripts/                 # CLI entrypoints
├── src/tta3dcache/          # Python package
│   ├── engine/              # Training/evaluation loops
│   ├── models/              # Model definitions
│   └── utils/               # Utilities
├── tests/                   # Unit tests
└── requirements.txt         # Core dependencies
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --help
```

## Next steps

- Add your dataset and dataloaders under `src/tta3dcache/`.
- Add model variants under `src/tta3dcache/models/`.
- Put experiment settings in `configs/`.
- Add reproducible tests under `tests/`.
