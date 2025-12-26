# Helper Scripts

Local Python scripts for development tasks. Run with `.venv` environment.

## Setup

### Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## Available Scripts

### 1. Generate Reports

Generate markdown reports from MLflow runs:

```powershell
# Latest run
python scripts/generate_report.py --latest

# Specific run
python scripts/generate_report.py --run-id abc123...

# Experiment summary
python scripts/generate_report.py --summary --top-n 10
```

**Output**: Markdown reports in `reports/` directory

### 2. Download Data (via Kaggle API)

```powershell
# Via workflow
.agent/workflows/download-data.md

# Or directly
kaggle competitions download -c george-b-moody-physionet-challenge-2024 -p data/raw
```

## When to Use .venv vs Docker

### Use .venv for:
- ✅ Generating MLflow reports
- ✅ Downloading Kaggle data
- ✅ Data exploration in Jupyter
- ✅ Quick utility scripts
- ✅ MLflow UI access scripts

### Use Docker for:
- ✅ Model training (baseline or SignalSavants)
- ✅ Hyperparameter tuning with Ray
- ✅ Inference and submission generation
- ✅ Full pipeline execution

## Requirements

The `.venv` has lightweight dependencies from `requirements-local.txt`:
- MLflow for experiment tracking
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- Hydra/OmegaConf for config
- Kaggle API for data download
- Jupyter for notebooks

**No heavy ML dependencies** (PyTorch, nnU-Net, Ray) - those are in Docker only.
