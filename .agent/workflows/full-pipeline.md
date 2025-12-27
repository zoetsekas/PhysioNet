---
description: Full end-to-end pipeline from training to Kaggle submission
---

# Full Pipeline: Train → Export → Submit

This master workflow chains together the complete ECG digitization pipeline.

## Overview

1. `/health-check` - Verify system requirements
2. `/train-signalsavants` or `/train-ray` - Train the model
3. `/extract-best-params` - Get best hyperparameters from MLflow
4. `/export-model` - Export model to git repo
5. Push to GitHub → Kaggle notebook runs inference

## Quick Start

### Option A: Standard Training Pipeline

1. First, run health check:
   See: `/health-check`

2. Train the model:
   See: `/train-signalsavants`

3. Export best model to repo:
   See: `/export-model`

### Option B: Ray Training with Tuning

1. Run health check:
   See: `/health-check`

2. Run hyperparameter tuning:
   See: `/tune-ray`

3. Extract best parameters and update configs:
   See: `/extract-best-params`

4. Train final model with best params:
   See: `/train-ray`

5. Export model:
   See: `/export-model`

## One-Shot Commands

### Quick Train (5 epochs for testing)

```powershell
# Health check
docker info; curl.exe http://localhost:5050/health; nvidia-smi

# Train
docker run --gpus all -it --rm `
  --network trading_network `
  --shm-size=16g `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/reports:/app/reports `
  -v ${PWD}/src:/app/src `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train approach=signalsavants training.epochs=5

# Export model
Copy-Item models/checkpoints/best_model.pt models/exports/ecg_digitizer.pt
git add models/exports/
git commit -m "Add trained model"
git push
```

### Production Train (full epochs)

```powershell
docker run --gpus all -it --rm `
  --network trading_network `
  --shm-size=16g `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/reports:/app/reports `
  -v ${PWD}/src:/app/src `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train approach=signalsavants training.epochs=100
```

## Post-Training Checklist

- [ ] Check MLflow UI: http://localhost:5050
- [ ] Review training metrics and visualizations
- [ ] Export best model to `models/exports/`
- [ ] Push to GitHub
- [ ] Update Kaggle notebook dataset reference
- [ ] Run Kaggle notebook for submission

## Kaggle Submission

After pushing the model to GitHub:

1. Go to Kaggle
2. Create a new dataset from your GitHub repo (or update existing)
3. Open the `notebooks/kaggle_submission.ipynb` notebook
4. Attach your dataset
5. Run all cells
6. Submit `submission.parquet`

## Troubleshooting

**Training too slow?**
- Reduce `data.image_size` in config
- Reduce `training.epochs` for testing

**Out of memory?**
- Reduce `data.batch_size`
- Use `/train-ray` with smaller config

**MLflow not connecting?**
- Check `/health-check`
- Ensure `trading_network` exists
