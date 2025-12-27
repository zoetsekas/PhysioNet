---
description: Train ECG digitization model using baseline approach
---

# Train Model - Baseline Approach

Trains the ECG digitization model using the original baseline architecture (UNet++, skeleton tracing, SNR+MSE loss).

## Prerequisites Check

### 1. Check Docker Desktop is Running

```powershell
docker info
```

**Expected**: Docker info displays successfully  
**If fails**: Start Docker Desktop and wait for it to be ready

## Training Steps

### 3. Build Docker Image (if needed)

// turbo
```powershell
docker compose -f docker/docker-compose.yml build
```

### 4. Run Training with Baseline Approach

```powershell
docker run --gpus all -it --rm `
  --network trading_network `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/reports:/app/reports `
  -v ${PWD}/src:/app/src `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train approach=baseline
```

## Post-Training

### 5. Check Results in MLflow

Open MLflow UI: [http://localhost:5050](http://localhost:5050)

### 6. View Generated Report

```powershell
Get-ChildItem reports/ -Filter "run_*.md" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
```

## Expected Performance

- **Training Time**: ~2-3 hours
- **GPU Memory**: ~8 GB
- **Target SNR**: 15-18 dB

## Troubleshooting

**Docker not responding**:
```powershell
Restart-Service docker
```

**MLflow connection error**:
```powershell
# Check if MLflow is listening
netstat -an | Select-String "5050"
```

**Out of GPU memory**:
- Reduce batch size in `configs/data/default.yaml`
- Current: 4 → Try: 2