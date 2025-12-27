---
description: Train ECG model using Ray distributed training
---

# Train ECG Model with Ray Training

This workflow uses Ray Train for training with automatic checkpointing and MLflow integration on a single NVIDIA 5090 (32GB).

## Prerequisites

1. Build Docker image (if not already built):
   ```powershell
   docker compose -f docker/docker-compose.yml build ecg-digitization
   ```

2. Verify Docker Desktop is running:
   ```powershell
   docker info
   ```

3. Check MLflow server (from within Docker network):
   ```powershell
   docker run --rm --network trading_network curlimages/curl:latest http://mlflow-server:5050/health
   ```

4. Verify GPU availability:
   ```powershell
   nvidia-smi
   ```

## Training Steps

### 1. Ray Training (Single GPU - NVIDIA 5090 32GB)

Run training with Ray Train:

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
  python -m ecg_digitization.train_ray +mode=train
```

### 2. Training with Custom Config

Adjust training parameters:

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
  python -m ecg_digitization.train_ray +mode=train `
    training.epochs=50 `
    training.learning_rate=1e-4 `
    data.batch_size=4
```

### 3. Full Resolution Training (uses more GPU memory)

For higher resolution images with 32GB GPU memory:

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
  python -m ecg_digitization.train_ray +mode=train `
    data.image_size=[1024,1280] `
    data.batch_size=2 `
    training.epochs=100
```

## Monitor Training

### View MLflow UI

Open in browser: http://localhost:5050

### Check GPU Usage

```powershell
nvidia-smi -l 1
```

## Expected Output

- Model checkpoints saved to: `models/checkpoints/`
- Training metrics logged to MLflow
- Best model saved as: `models/checkpoints/best_model.pt`

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or image size:

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
  python -m ecg_digitization.train_ray +mode=train `
    data.batch_size=1 `
    data.image_size=[512,640]
```

### Ray Initialization Errors

Increase shared memory:
```powershell
--shm-size=24g
```
