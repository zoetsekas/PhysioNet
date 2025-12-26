# Docker Setup for ECG Digitization

This directory contains Docker configurations for the ECG Image Digitization project.

## Image Architecture

The project uses a **two-stage Docker image architecture**:

### 1. Base ML Image: `trading_base:latest`
**Source**: [`base_ml.Dockerfile`](base_ml.Dockerfile)

**Purpose**: Provides core ML/DL dependencies shared across multiple projects.

**Contents**:
- **Base OS**: Ubuntu 24.04
- **GPU Support**: NVIDIA CUDA 13.0.1 with cuDNN runtime
- **Python**: 3.12
- **Package Manager**: `uv` for fast dependency installation
- **Core ML Libraries**:
  - PyTorch with CUDA support
  - Ray (distributed computing)
  - RAPIDS.ai (GPU-accelerated data science)
  - scikit-learn
  - scipy, numpy, pandas

**Build Command**:
```bash
# From project root
docker build -f docker/base_ml.Dockerfile -t trading_base:latest .

# Or using Makefile
make docker-build-base
```

**Build Time**: ~15-30 minutes (depends on network speed and package compilation)

---

### 2. Application Image: `ecg-digitization:latest`
**Source**: [`Dockerfile`](Dockerfile)

**Purpose**: Extends the base image with ECG-specific dependencies and application code.

**Additional Dependencies**:
- **Image Processing**: OpenCV, Pillow, Albumentations
- **Signal Processing**: WFDB, NeuroKit2, BioSPPy
- **Models**: Transformers, TIMM, nnU-Net, Segmentation Models
- **Experiment Tracking**: MLflow, W&B, Optuna
- **Configuration**: Hydra, OmegaConf
- **Application Code**: Source from `src/` directory

**Build Command**:
```bash
# From project root (requires trading_base:latest to exist)
docker build -f docker/Dockerfile -t ecg-digitization:latest .

# Or using Makefile
make docker-build

# Or using docker-compose
docker-compose -f docker/docker-compose.yml build
```

**Build Time**: ~5-10 minutes (after base image is built)

---

## Quick Start

### First-Time Setup
```bash
# 1. Build base image (one-time or when base dependencies change)
make docker-build-base

# 2. Build application image
make docker-build

# 3. Run with docker-compose
docker-compose -f docker/docker-compose.yml up -d
```

### Development Workflow
```bash
# Open interactive shell
make docker-shell

# Run training
make docker-train

# Run Ray-based training
make docker-train-ray

# Run hyperparameter tuning
make docker-tune
```

---

## Docker Compose Services

The [`docker-compose.yml`](docker-compose.yml) defines two services:

### `ecg-digitization` (Main Service)
- **Purpose**: Main development container
- **GPU**: All GPUs available
- **Ports**: 
  - 8888 (Jupyter)
  - 5000 (MLflow)
  - 6006 (TensorBoard)
- **Volumes**: Mounts data, models, source code, configs, and logs

### `jupyter` (Jupyter Lab Service)
- **Purpose**: Interactive EDA and notebook development
- **GPU**: All GPUs available
- **Port**: 8889 → 8888 (internal)
- **Command**: Auto-starts Jupyter Lab without token authentication

---

## Environment Variables

Set these in your `.env` file or export them:

```bash
# Required for HuggingFace model downloads
export HUGGING_FACE_HUB_TOKEN=your_token_here

# Optional - for experiment tracking
export WANDB_API_KEY=your_wandb_key
export MLFLOW_TRACKING_URI=http://host.docker.internal:5050
```

---

## Rebuilding Images

### When to Rebuild Base Image
- Python version upgrade
- Core ML library updates (PyTorch, Ray, RAPIDS)
- CUDA version change
- Updates to `requirements_core.txt`

### When to Rebuild Application Image
- Application dependency changes
- Source code updates (if not using volume mounts)
- Configuration changes
- Updates to `requirements.txt`

---

## Troubleshooting

### Base Image Build Failures
```bash
# Increase timeout for large package downloads
export UV_HTTP_TIMEOUT=600

# Rebuild without cache
docker build --no-cache -f docker/base_ml.Dockerfile -t trading_base:latest .
```

### Application Image Missing Base
```bash
# Error: "failed to solve with frontend dockerfile.v0: failed to create LLB definition..."
# Solution: Build base image first
make docker-build-base
```

### GPU Not Available
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 nvidia-smi

# Check Docker daemon configuration
# Ensure nvidia-container-toolkit is installed
```

---

## File Reference

| File                                             | Purpose                             |
| ------------------------------------------------ | ----------------------------------- |
| [`base_ml.Dockerfile`](base_ml.Dockerfile)       | Multi-stage build for base ML image |
| [`Dockerfile`](Dockerfile)                       | Application image extending base    |
| [`docker-compose.yml`](docker-compose.yml)       | Service orchestration               |
| [`requirements_core.txt`](requirements_core.txt) | Base image Python dependencies      |
| [`requirements.txt`](requirements.txt)           | Application-specific dependencies   |

---

## Best Practices

1. **Layer Caching**: Base image rarely changes, so it's cached between application rebuilds
2. **Volume Mounts**: Source code is mounted for live development without rebuilds
3. **Non-root User**: Container runs as `trading` user for security
4. **GPU Isolation**: Use NVIDIA runtime for GPU access
5. **Shared Memory**: Configured with 8GB `shm_size` for PyTorch DataLoader workers
