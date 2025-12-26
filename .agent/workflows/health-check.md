---
description: Quick health check for Docker Desktop and MLflow server
---

# System Health Check

Verifies that all required services are running before training.

## Docker Desktop

### 1. Check Docker Service

// turbo
```powershell
docker info
```

**Expected Output**: Docker version, server info  
**Status**: ✅ Docker is running

**If fails**:
- Start Docker Desktop application
- Wait 30-60 seconds for initialization
- Run check again

### 2. Check Docker Compose

// turbo
```powershell
docker compose version
```

**Expected Output**: Docker Compose version  
**Status**: ✅ Docker Compose is available

## MLflow Server

### 3. Check MLflow Health Endpoint

```powershell
curl http://localhost:5050/health
```

**Expected Output**: HTTP 200 OK  
**Status**: ✅ MLflow server is healthy

**If fails**:

Start MLflow server:
```powershell
mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Or use Docker:
```powershell
docker run -d -p 5050:5000 --name mlflow-server `
  -v ${PWD}/mlruns:/mlflow/mlruns `
  ghcr.io/mlflow/mlflow:latest `
  mlflow server --host 0.0.0.0 --backend-store-uri /mlflow/mlruns
```

### 4. Check MLflow UI Access

Open browser: [http://localhost:5050](http://localhost:5050)

**Expected**: MLflow UI loads  
**Status**: ✅ MLflow UI is accessible

## GPU Availability

### 5. Check NVIDIA GPU

```powershell
nvidia-smi
```

**Expected Output**:
- GPU: NVIDIA GeForce RTX 5090
- Memory: 24GB+ available
- Driver version displayed

**Status**: ✅ GPU is available

**If fails**:
- Update NVIDIA drivers
- Restart Docker Desktop
- Check Docker GPU settings

### 6. Check Docker GPU Access

```powershell
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Expected**: Same output as host `nvidia-smi`  
**Status**: ✅ Docker can access GPU

## Project Structure

### 7. Verify Data Directory

```powershell
Test-Path data/
```

**Expected**: True  
**Status**: ✅ Data directory exists

### 8. Verify Models Directory

```powershell
Test-Path models/checkpoints/
```

**Expected**: True  
**Status**: ✅ Models directory exists

### 9. Verify Reports Directory

```powershell
Test-Path reports/
```

**Expected**: True  
**Status**: ✅ Reports directory exists

## Summary

If all checks pass:
- ✅ **System is ready for training**
- Proceed with workflow: `train-baseline` or `train-signalsavants` or `tune`

If any checks fail:
- ❌ **Fix the failed component before proceeding**
- See troubleshooting steps above
