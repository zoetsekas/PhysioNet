---
description: Extract best hyperparameters from MLflow and update configs
---

# Extract Best Hyperparameters

After running hyperparameter tuning, use this workflow to extract the best parameters from MLflow and update your config files.

## Prerequisites

1. MLflow server must be running with tuning results:
   ```powershell
   curl.exe http://localhost:5050/health
   ```

## Usage

### 1. View Best Parameters (Read Only)

```powershell
docker run --rm `
  --network trading_network `
  -v ${PWD}/src:/app/src `
  ecg-digitization:latest `
  python -m ecg_digitization.utils.extract_best_params
```

### 2. View as JSON (for scripting)

```powershell
docker run --rm `
  --network trading_network `
  -v ${PWD}/src:/app/src `
  ecg-digitization:latest `
  python -m ecg_digitization.utils.extract_best_params --json
```

### 3. Preview Config Updates (Dry Run)

```powershell
docker run --rm `
  --network trading_network `
  -v ${PWD}/src:/app/src `
  -v ${PWD}/configs:/app/configs `
  ecg-digitization:latest `
  python -m ecg_digitization.utils.extract_best_params `
    --update-config /app/configs/training/default.yaml
```

### 4. Apply Config Updates

```powershell
docker run --rm `
  --network trading_network `
  -v ${PWD}/src:/app/src `
  -v ${PWD}/configs:/app/configs `
  ecg-digitization:latest `
  python -m ecg_digitization.utils.extract_best_params `
    --update-config /app/configs/training/default.yaml `
    --apply
```

## Options

| Option                | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `-e, --experiment`    | MLflow experiment name (default: ecg-digitization)     |
| `-u, --tracking-uri`  | MLflow server URI (default: http://mlflow-server:5050) |
| `-m, --metric`        | Metric to optimize (default: best_val_snr)             |
| `--minimize`          | Minimize metric instead of maximize                    |
| `-c, --update-config` | Path to config file to update                          |
| `--apply`             | Apply changes (otherwise dry run)                      |
| `--json`              | Output as JSON                                         |

## Example Output

```
🏆 Best Parameters from 'ecg-digitization'
   Run ID: abc123def456
   best_val_snr: 15.234

Hyperparameters:
  learning_rate: 0.0001
  batch_size: 4
  hidden_dim: 256
  weight_decay: 1e-05
  encoder_name: resnet50
```

## Workflow After Tuning

1. Run hyperparameter tuning: `/tune-ray`
2. Extract best params: This workflow
3. Review and apply to configs
4. Train final model with optimized params: `/train-signalsavants`
