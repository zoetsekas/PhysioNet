---
description: Download Kaggle competition data (uses local .venv)
---

# Download Competition Data

Downloads the PhysioNet ECG Digitization competition data from Kaggle using the local Python environment.

## Prerequisites

### 1. Kaggle API Credentials

Ensure you have `~/.kaggle/kaggle.json` with your API credentials.

If not, create it:
```powershell
mkdir $env:USERPROFILE\.kaggle
# Copy your kaggle.json to this directory
```

### 2. Activate Local Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## Download Data

### 3. Download Competition Files

// turbo
```powershell
kaggle competitions download -c george-b-moody-physionet-challenge-2024 -p data/raw
```

### 4. Extract Data

```powershell
Expand-Archive data/raw/*.zip -DestinationPath data/raw/
```

### 5. Verify Download

```powershell
Get-ChildItem data/raw/
```

**Expected**:
- `training/` directory with ECG images
- `test/` directory with test images
- CSV files with metadata

## Alternative: Manual Download

If Kaggle API fails:

1. Visit: https://www.kaggle.com/competitions/george-b-moody-physionet-challenge-2024/data
2. Download manually
3. Extract to `data/raw/`

## Data Structure

After extraction:
```
data/
└── raw/
    ├── training/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    ├── test/
    │   └── ...
    ├── train.csv
    └── test.csv
```

## Next Steps

After data download:
- Run baseline training: `/.agent/workflows/train-baseline.md`
- Or SignalSavants training: `/.agent/workflows/train-signalsavants.md`
