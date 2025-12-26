# ECG Digitization System Architecture

## Overview

This document describes the system architecture for the PhysioNet ECG Image Digitization competition solution.

## Problem Statement

Extract 12-lead ECG time-series signals from paper printout images (scans and photographs).

### Challenges
- **Image Variability**: Digital scans vs mobile photos with different quality
- **Artifacts**: Rotation, blur, noise, perspective distortion
- **Physical Damage**: Stains, mold, tears, fading on paper
- **Grid Removal**: Need to extract signal lines while removing background grid

### Evaluation
- **Metric**: Signal-to-Noise Ratio (SNR) in decibels
- **Alignment**: Signals are aligned with ground truth before scoring (±0.2s shift, vertical offset)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│                    ECG Image (JPG/PNG)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Resize     │→ │ Augmentation │→ │ Normalize    │         │
│  │ 1024×1280    │  │ (train only) │  │ (ImageNet)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL PIPELINE                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    IMAGE ENCODER                          │ │
│  │           ResNet50 / Swin-B / ConvNeXt                   │ │
│  │              (pretrained on ImageNet)                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                  FEATURE DECODER                          │ │
│  │                    UNet++ / FPN                           │ │
│  │           Output: [B, 64, H/4, W/4]                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                 SIGNAL REGRESSION HEAD                    │ │
│  │  Height Pooling → Conv1D Decoder → 12 Lead Outputs       │ │
│  │              Output: [B, 12, T]                           │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
│              12 Lead ECG Signals (Time Series)                 │
│     I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Pipeline

**Dataset Class** (`ECGImageDataset`):
- Loads ECG images and ground truth signals
- Supports train/test modes
- Handles variable image formats

**Transforms**:
- Training: Resize, rotation, perspective, blur, noise, color jitter
- Validation: Resize, normalize only

**Preprocessing** (`preprocessing.py`):
- Grid detection and removal
- Signal line enhancement
- Lead region detection

### 2. Model Architecture

**Encoder** (via `timm` or HuggingFace):
- ResNet50 (baseline)
- Swin Transformer (advanced)
- ConvNeXt (efficient)

**Decoder** (via `segmentation-models-pytorch`):
- UNet++ for multi-scale feature fusion
- FPN for pyramid features

**Signal Head**:
- Height pooling: [B, C, H, W] → [B, C, W]
- Conv1D decoder for temporal modeling
- Output projection to 12 leads

### 3. Training

**Loss Function**:
```python
loss = snr_weight * SNR_loss + mse_weight * MSE_loss
```
- SNR loss matches competition metric
- MSE provides gradient stability

**Optimizer**: AdamW with cosine annealing

**Hardware**: NVIDIA GPU with mixed precision training

### 4. Inference

**Predictor**:
- Loads trained model
- Runs batch inference
- Generates submission file in parquet format

**Test-Time Augmentation** (optional):
- Horizontal flip
- Slight rotation
- Brightness adjustment

## Docker Setup

Base image: `trading_base:latest` (contains PyTorch, Ray, RAPIDS.ai)

Additional dependencies:
- OpenCV, Pillow, albumentations (image processing)
- wfdb, neurokit2 (ECG signal processing)
- transformers, timm, segmentation-models-pytorch (models)

MLflow tracking: http://localhost:5050 (external)

## Kaggle Submission Constraints

- GPU runtime ≤ 9 hours
- No internet access
- All models must be pre-packaged as Kaggle Dataset

## Directory Structure

```
src/ecg_digitization/
├── data/
│   ├── dataset.py          # ECGImageDataset
│   ├── transforms.py       # Albumentations transforms
│   ├── preprocessing.py    # Grid removal, signal extraction
│   ├── hough_deskew.py     # Hough Transform deskewing (NEW)
│   └── ray_data.py         # Ray parallel data loading
├── models/
│   ├── backbone.py         # timm/HuggingFace backbones
│   ├── encoder_decoder.py  # UNet++/FPN decoder
│   ├── signal_head.py      # 2D→1D signal regression
│   ├── nnunet_segmenter.py # nnU-Net integration (NEW)
│   └── digitizer.py        # Full model combining all parts
├── training/
│   ├── trainer.py          # Training loop
│   ├── losses.py           # SNR loss, combined loss
│   ├── segmentation_loss.py # Dice, Focal, Tversky (NEW)
│   └── ray_trainer.py      # Ray distributed training
├── inference/
│   ├── predictor.py        # Inference and submission
│   └── vectorizer.py       # Column-wise extraction (NEW)
├── utils/
│   ├── config.py           # Hydra config utilities
│   ├── logging.py          # Loguru setup
│   ├── metrics.py          # SNR computation
│   └── calibration.py      # Multi-method calibration (NEW)
└── pipeline_factory.py     # Configurable pipeline builder (NEW)
```

## Configurable Approach System

The system now supports **A/B testing** between two approaches:

### Approach Selection

Configure via Hydra:
```bash
# Use SignalSavants (winner approach)
python -m ecg_digitization.train approach=signalsavants

# Use baseline (original approach)
python -m ecg_digitization.train approach=baseline
```

### Configuration Structure

```
configs/
├── approach/
│   ├── default.yaml        # Default (SignalSavants)
│   ├── signalsavants.yaml  # PhysioNet 2024 winner
│   └── baseline.yaml       # Original baseline
```

### Comparison Matrix

| Component         | Baseline         | SignalSavants            | Configurable                 |
| ----------------- | ---------------- | ------------------------ | ---------------------------- |
| **Preprocessing** | Resize/Normalize | Hough Deskewing          | `preprocessing.deskew`       |
| **Segmentation**  | UNet++           | nnU-Net                  | `segmentation.model`         |
| **Extraction**    | Skeleton Tracing | Column-wise Centroid     | `extraction.method`          |
| **Loss Function** | SNR + MSE        | Dice + CE                | `loss.type`                  |
| **Calibration**   | Grid + Pulse     | Grid + Pulse + Blind QRS | `calibration.fallback_chain` |

### Pipeline Factory

The `PipelineFactory` class dynamically builds components:

```python
from ecg_digitization.pipeline_factory import create_pipeline_from_config

factory = create_pipeline_from_config(config)

# Create components based on approach
preprocessor = factory.create_preprocessor()  # None or HoughDeskewer
segmenter = factory.create_segmenter()        # UNet++ or nnU-Net
loss_fn = factory.create_loss()              # Regression or Segmentation
extraction_method = factory.get_extraction_method()  # "skeleton" or "column_wise"
```

### Example Configurations

**SignalSavants (Winner)**:
```yaml
method: "signalsavants"
preprocessing:
  deskew: true
  deskew_method: "hough"
segmentation:
  model: "nnunet"
extraction:
  method: "column_wise"
loss:
  type: "segmentation"
calibration:
  fallback_chain: ["pulse", "grid", "blind"]
```

**Baseline**:
```yaml
method: "baseline"
preprocessing:
  deskew: false
segmentation:
  model: "unet++"
extraction:
  method: "skeleton"
loss:
  type: "regression"
calibration:
  fallback_chain: ["grid", "pulse"]
```

### Performance Expectations

| Approach          | Expected SNR | Training Time | GPU Memory |
| ----------------- | ------------ | ------------- | ---------- |
| **Baseline**      | 15-18 dB     | ~2-3 hours    | 8GB        |
| **SignalSavants** | 20-25 dB     | ~4-6 hours    | 12GB       |

The configurable system enables:
- **Comparative benchmarking** between approaches
- **Gradual migration** from baseline to advanced methods
- **Ablation studies** to measure impact of each component
