---
description: Run hyperparameter tuning with Ray Tune and Optuna
---

# Hyperparameter Tuning

Runs automated hyperparameter search using Ray Tune with Optuna and ASHA early stopping.

## Prerequisites Check

### 1. Check Docker Desktop is Running

```powershell
docker info
```

**Expected**: Docker info displays successfully  
**If fails**: Start Docker Desktop and wait for it to be ready

### 2. Check MLflow Server is Running

```powershell
curl.exe http://localhost:5050/health
```

**Expected**: Returns HTTP 200  
**If fails**: Start MLflow server with `mlflow server --host 0.0.0.0 --port 5050`

### 3. Verify GPU Availability and Memory

```powershell
nvidia-smi
```

**Expected**: RTX 5090 with 32GB memory (single GPU)  
**Note**: Tuning runs trials sequentially on single GPU

## Configuration

### 4. Review Tuning Settings

```powershell
Get-Content configs/ray.yaml
```

Key parameters:
- `tune.num_samples`: 10 trials (sequential)
- `tune.max_concurrent_trials`: 1 (single GPU)
- `tune.grace_period`: 3 epochs minimum

**Adjust if needed**: Edit `configs/ray.yaml`

## Tuning Steps

### 5. Build Docker Image (if needed)

// turbo
```powershell
docker compose -f docker/docker-compose.yml build
```

### 6. Run Hyperparameter Tuning

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
  python -m ecg_digitization.train_ray +mode=tune
```

**Estimated Time**: 4-6 hours (10 trials × ~30min each, sequential)

## Monitoring

### 7. Monitor Progress in MLflow

Open: [http://localhost:5050](http://localhost:5050)

You should see multiple nested runs under the main tuning run.

### 8. Watch Ray Dashboard (Optional)

If Ray dashboard is exposed:
```powershell
# Check Ray status
docker exec -it ecg-digitization ray status
```

## Post-Tuning

### 9. View Best Hyperparameters

Check the parent run in MLflow:
- Best parameters logged with `best_` prefix
- Best validation SNR logged as `best_val_snr`

### 10. Generate Tuning Report

```powershell
docker run -it --rm `
  --network trading_network `
  -v ${PWD}/reports:/app/reports `
  ecg-digitization:latest `
  python -c "from ecg_digitization.utils import ExperimentReportGenerator; g = ExperimentReportGenerator('http://mlflow-server:5050'); g.generate_experiment_summary('ecg-digitization', 20)"
```

### 11. Train Final Model with Best Hyperparameters

Extract best params from MLflow, then:

```powershell
docker run --gpus all -it --rm `
  --network trading_network `
  --shm-size=8g `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/src:/app/src `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train approach=signalsavants `
    training.learning_rate=<best_lr> `
    training.weight_decay=<best_wd>
```

## Search Space

Default search space (defined in `training/ray_trainer.py`):
- `learning_rate`: [1e-5, 1e-3] (log scale)
- `weight_decay`: [1e-6, 1e-3] (log scale)
- `batch_size`: [2, 4, 8]

## Expected Results

- **Number of Trials**: 10 (sequential)
- **Completion Rate**: ~60-80% (ASHA stops poor performers)
- **Best SNR Improvement**: +2-5 dB over defaults
- **Resource Usage**: Single GPU (NVIDIA 5090 32GB)

## Troubleshooting

**Tuning crashes mid-way**:
```powershell
# Resume from checkpoint
docker run --gpus all -it --rm `
  --network trading_network `
  --shm-size=16g `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/src:/app/src `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train_ray +mode=tune +resume=true
```

**Out of GPU memory**:
- Reduce `tune.max_concurrent_trials` to 1
- Reduce search space batch sizes

**Ray worker timeout**:
- Increase `ray.num_cpus` in `configs/ray.yaml`
- Check Docker resource limits

**MLflow logging slow**:
- Reduce `tune.num_samples` to 10
- Increase `tune.grace_period` to stop poor trials faster
