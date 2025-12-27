---
description: Train ECG digitization model using SignalSavants approach
---

# Train Model - SignalSavants Approach

Trains the ECG digitization model using the PhysioNet 2024 Challenge winning architecture (nnU-Net, Hough deskewing, column-wise vectorization, Dice+CE loss).

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

### 3. Verify GPU Availability

```powershell
nvidia-smi
```

**Expected**: Shows RTX 5090 with available memory  
**If fails**: Check NVIDIA drivers, restart Docker Desktop

## Training Steps

### 4. Build Docker Image (if needed)

// turbo
```powershell
docker compose -f docker/docker-compose.yml build
```

### 5. Run Training with SignalSavants Approach

```powershell
docker run --gpus all -it --rm `
  --network trading_network `
  --shm-size=8g `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/reports:/app/reports `
  -v ${PWD}/src:/app/src `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train approach=signalsavants
```

**Note**: `--shm-size=8g` is required for Ray's shared memory

## Alternative: Ray Distributed Training

For better performance with multi-GPU support:

```powershell
docker run --gpus all -it --rm `
  --network trading_network `
  --shm-size=8g `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/reports:/app/reports `
  -v ${PWD}/src:/app/src `
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5050 `
  ecg-digitization:latest `
  python -m ecg_digitization.train_ray mode=train
```

## Post-Training

### 6. Check Results in MLflow

Open MLflow UI: [http://localhost:5050](http://localhost:5050)

Look for run tagged with `approach=signalsavants`

### 7. View Generated Report

```powershell
Get-ChildItem reports/ -Filter "run_*.md" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
```

### 8. Compare with Baseline

Generate experiment summary:

```powershell
docker run -it --rm `
  --network trading_network `
  -v ${PWD}/reports:/app/reports `
  ecg-digitization:latest `
  python -c "from ecg_digitization.utils import ExperimentReportGenerator; g = ExperimentReportGenerator('http://mlflow-server:5050'); g.generate_experiment_summary('ecg-digitization', 10)"
```

## Expected Performance

- **Training Time**: ~4-6 hours
- **GPU Memory**: ~12 GB
- **Target SNR**: 20-25 dB (synthetic), 18-22 dB (real scans)

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
- Disable nnU-Net: `approach.segmentation.model=unet++`

**nnU-Net initialization fails**:
- Check Docker logs: `docker logs ecg-digitization`
- System will automatically fallback to UNet++