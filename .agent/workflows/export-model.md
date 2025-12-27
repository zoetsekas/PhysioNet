---
description: Export best model to git repo for Kaggle submission
---

# Export Best Model to Git Repository

This workflow exports the best trained model to the git repository so the Kaggle notebook can load it for inference.

## Prerequisites

1. Training completed with best model saved
2. Git LFS installed (for large model files)

## Steps

### 1. Check if best model exists

```powershell
Test-Path models/checkpoints/best_model.pt
```

### 2. Setup Git LFS for model files

```powershell
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.safetensors"
git add .gitattributes
```

### 3. Create models export directory

```powershell
New-Item -ItemType Directory -Force -Path models/exports
```

### 4. Copy best model to exports

```powershell
Copy-Item models/checkpoints/best_model.pt models/exports/ecg_digitizer.pt
```

### 5. Create model metadata

```powershell
docker run --rm `
  --network trading_network `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/src:/app/src `
  ecg-digitization:latest `
  python -c "
import torch
import json
from pathlib import Path

# Load model checkpoint
checkpoint = torch.load('/app/models/exports/ecg_digitizer.pt', map_location='cpu')

# Extract metadata
metadata = {
    'architecture': 'ECGDigitizer',
    'encoder': 'resnet50',
    'image_size': [512, 640],
    'num_leads': 12,
    'signal_length': 5000,
}

# If checkpoint has epoch/loss info
if isinstance(checkpoint, dict):
    metadata['epoch'] = checkpoint.get('epoch', 'unknown')
    metadata['val_loss'] = checkpoint.get('val_loss', 'unknown')

# Save metadata
Path('/app/models/exports/model_metadata.json').write_text(json.dumps(metadata, indent=2))
print('Metadata saved:', metadata)
"
```

### 6. Verify export

```powershell
Get-ChildItem models/exports/
```

### 7. Add to git

```powershell
git add models/exports/
git commit -m "Add best trained model for inference"
```

### 8. Push to remote

```powershell
git push
```

## Model Usage in Kaggle

The Kaggle notebook will:
1. Clone the repo or access model via Kaggle datasets
2. Load the model: `model.load_state_dict(torch.load('models/exports/ecg_digitizer.pt'))`
3. Run inference on test images
4. Generate submission.csv

## Notes

- Git LFS is required for files over 100MB
- Alternatively, upload model to Kaggle as a private dataset
- Model file size: Check with `(Get-Item models/exports/ecg_digitizer.pt).Length / 1MB`
