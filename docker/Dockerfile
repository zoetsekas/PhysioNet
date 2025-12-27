# ECG Digitization Docker Image
# Base image with PyTorch, Ray, RAPIDS.ai, sklearn, scipy, pandas
FROM trading_base:latest

LABEL maintainer="zoetsekas@outlook.com"
LABEL description="ECG Image Digitization - PhysioNet Kaggle Competition"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# nnU-Net environment variables (required for nnunetv2)
ENV nnUNet_raw="/app/nnunet/raw" \
    nnUNet_preprocessed="/app/nnunet/preprocessed" \
    nnUNet_results="/app/nnunet/results"

# Ray configuration
ENV RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Set working directory
WORKDIR /app

USER root

# Install system dependencies for image processing and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml README.md ./

# Install Python dependencies
# Core image and signal processing libraries
RUN pip install \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    albumentations>=1.3.0 \
    wfdb>=4.1.0 \
    neurokit2>=0.2.0 \
    biosppy>=1.0.0

# ML and model libraries
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    torchvision \
    transformers>=4.35.0 \
    timm>=0.9.0 \
    segmentation-models-pytorch>=0.3.0

# Data handling and utilities
RUN pip install \
    pandas>=2.2.0 \
    pyarrow>=14.0.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    loguru>=0.7.0 \
    rich>=13.0.0

# Experiment tracking
RUN pip install \
    wandb>=0.16.0 \
    mlflow>=2.9.0 \
    optuna>=3.4.0 \
    nnunetv2>=2.2.1 \
    scikit-image>=0.21.0 \
    SimpleITK>=2.3.0 \
    scipy>=1.11.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    plotly>=5.18.0

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Install the package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/data/train \
    /app/data/test \
    /app/data/submissions \
    /app/models/checkpoints \
    /app/models/exports \
    /app/logs \
    /app/nnunet/raw \
    /app/nnunet/preprocessed \
    /app/nnunet/results

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Expose port for MLflow/Jupyter if needed
EXPOSE 8888 5000

# Default command
CMD ["python", "-m", "ecg_digitization.train"]
