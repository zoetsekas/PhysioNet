---
description: Apply tuning results and train final model
---

# Post-Tuning Workflow

After running `/tune-ray`, use this workflow to apply the best hyperparameters and train the final model.

## 1. Extract Best Parameters

Extract the best hyperparameters from the MLflow tuning experiment and update the local configuration file.

```powershell
docker run -it --rm `
  --network trading_network `
  -v ${PWD}/src:/app/src `
  -v ${PWD}/configs:/app/configs `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.utils.extract_best_params --update-config configs/training/default.yaml
```

## 2. Train Final Model

Train the final model using the updated configuration (and Ray for distributed training if configured, though single-GPU training is often sufficient).

```powershell
docker run --gpus all -it --rm `
  --network trading_network `
  --shm-size=16g `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/reports:/app/reports `
  -v ${PWD}/src:/app/src `
  -v ${PWD}/logs:/app/logs `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train_ray
```

## 3. Export Model

Export the best checkpoint to the git repository.

```powershell
# Run the export-model workflow
# See: .agent/workflows/export-model.md
```
