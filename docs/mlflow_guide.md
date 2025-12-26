# MLflow Integration Guide

## Overview

The project now has comprehensive MLflow integration for experiment tracking, model management, and visualization.

## Features

### 1. **Experiment Tracking**
- Automatic logging of all hyperparameters
- Real-time metric tracking (training/validation loss, SNR)
- Run tagging (approach, model, project)

### 2. **Model Management**
- Automatic model artifact logging
- Model registry integration
- Best model checkpointing

### 3. **Visualizations**
- Training/validation loss curves
- SNR per lead bar charts
- Signal comparison plots (ground truth vs prediction)

## Quick Start

### Accessing MLflow UI

```bash
# MLflow server is running at:
http://localhost:5050
```

### Training with MLflow Tracking

```bash
# Standard training (automatic MLflow logging)
python -m ecg_digitization.train approach=signalsavants

# Ray training (automatic MLflow logging)
python -m ecg_digitization.train_ray mode=train
```

## What Gets Logged

### Parameters
- All Hydra configuration (nested configs flattened)
- Approach method (baseline vs signalsavants)
- Model architecture (encoder, decoder, etc.)
- Training hyperparameters (learning rate, batch size, etc.)

### Metrics (per epoch)
- `train_loss` - Training loss
- `val_loss` - Validation loss
- `val_snr` - Validation SNR (signal-to-noise ratio)
- `best_val_loss` - Best validation loss achieved

### Artifacts
- `best_model.pt` - Best model checkpoint
- `loss_curves.png` - Training/validation loss visualization
- `final_model/` - Registered PyTorch model

### Tags
- `approach` - baseline or signalsavants
- `model` - Encoder name (resnet50, swin, etc.)
- `project` - physionet-ecg-2024

## MLflow API Usage

### Programmatic Access

```python
from ecg_digitization.utils import MLflowExperimentTracker

# Initialize tracker
tracker = MLflowExperimentTracker(
    tracking_uri="http://localhost:5050",
    experiment_name="ecg-digitization",
    run_name="my_experiment",
    tags={"approach": "signalsavants"},
)

# Start run
tracker.start_run()

# Log parameters
tracker.log_config({
    "learning_rate": 0.001,
    "batch_size": 16,
})

# Log metrics
tracker.log_metric("train_loss", 0.5, step=1)
tracker.log_metrics({
    "val_loss": 0.3,
    "val_snr": 18.5,
}, step=1)

# Log model
tracker.log_model(
    model=my_model,
    artifact_path="model",
    registered_model_name="ecg_digitizer_v1",
)

# Log visualizations
from ecg_digitization.utils import create_loss_plot

loss_fig = create_loss_plot(train_losses, val_losses)
tracker.log_figure(loss_fig, "loss_curves.png")

# End run
tracker.end_run(status="FINISHED")
```

## Comparing Experiments

### Via UI

1. Navigate to [http://localhost:5050](http://localhost:5050)
2. Select "ecg-digitization" experiment
3. Click on multiple runs
4. Click "Compare" button
5. View side-by-side metrics and parameters

### Via Python

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5050")

# Search runs
runs = mlflow.search_runs(
    experiment_names=["ecg-digitization"],
    filter_string="tags.approach = 'signalsavants'",
    order_by=["metrics.val_snr DESC"],
)

print(runs[["run_id", "metrics.val_snr", "params.model.encoder_name"]])
```

## Model Registry

### Registering Models

Models are automatically registered during training:

```python
# Automatic registration (in train.py)
mlflow_tracker.log_model(
    model=model,
    artifact_path="final_model",
    registered_model_name=f"ecg_digitizer_{cfg.approach.method}",
)
```

### Loading Registered Models

```python
import mlflow.pytorch

# Load latest version
model_uri = "models:/ecg_digitizer_signalsavants/latest"
model = mlflow.pytorch.load_model(model_uri)

# Load specific version
model_uri = "models:/ecg_digitizer_signalsavants/1"
model = mlflow.pytorch.load_model(model_uri)
```

## Visualizations

### Loss Curves

Automatically generated and logged:
- Blue line: Training loss
- Red line: Validation loss
- Grid for readability

### SNR Per Lead

Bar chart showing SNR for each of the 12 leads:
- Green bars: SNR > 20 dB (excellent)
- Orange bars: 15-20 dB (good)
- Red bars: < 15 dB (needs improvement)

### Signal Comparison

Three-panel visualization:
1. Ground truth signal
2. Predicted signal
3. Overlay with error shading

## Experiment Organization

### Recommended Naming Convention

```bash
# Format: {approach}_{model}_{date}_{variant}
python -m ecg_digitization.train \
    approach=signalsavants \
    model.encoder_name=resnet50 \
    mlflow.run_name="signalsavants_resnet50_2024_baseline"
```

### Tagging Strategy

Use tags for filtering and organization:

```python
tags = {
    "approach": "signalsavants",  # or "baseline"
    "model": "resnet50",
    "augmentation": "heavy",
    "dataset_size": "50k",
    "gpu": "rtx5090",
}
```

## Best Practices

### 1. Always Set Run Names
```bash
python -m ecg_digitization.train \
    mlflow.run_name="descriptive_name_here"
```

### 2. Use Tags for Ablation Studies
```python
# Study 1: Baseline
tags = {
    "study": "ablation_preprocessing",
    "deskew": "false",
}

# Study 2: With deskewing
tags = {
    "study": "ablation_preprocessing",
    "deskew": "true",
}
```

### 3. Log Custom Metrics
```python
# In your training loop
if epoch % 5 == 0:
    snr_per_lead = compute_snr_per_lead(predictions, targets)
    for lead, snr in snr_per_lead.items():
        tracker.log_metric(f"snr_{lead}", snr, step=epoch)
```

### 4. Save Important Artifacts
```python
# Log config files
tracker.log_artifact("configs/config.yaml", "configs")

# Log sample predictions
tracker.log_artifact("outputs/predictions.csv", "results")
```

## Troubleshooting

### MLflow Server Not Running

```bash
# Check if server is running
curl http://localhost:5050

# If not running, start it
mlflow server --host 0.0.0.0 --port 5050
```

### Runs Not Appearing

Check that tracking URI is correct:
```python
import mlflow
print(mlflow.get_tracking_uri())
# Should print: http://localhost:5050
```

### From Docker Container

Use host address:
```yaml
# In config.yaml
mlflow:
  tracking_uri: "http://host.docker.internal:5050"
```

## Advanced Usage

### Custom Visualizations

```python
import matplotlib.pyplot as plt

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(custom_data)
ax.set_title("Custom Metri")
ax.grid(True)

# Log to MLflow
tracker.log_figure(fig, "custom_metric.png")
```

### Nested Runs

```python
# Parent run
tracker.start_run(run_name="hyperparameter_search")

for lr in [0.001, 0.01, 0.1]:
    # Child run
    child_tracker = MLflowExperimentTracker(...)
    child_tracker.start_run(
        run_name=f"lr_{lr}",
        nested=True,
    )
    
    # Train with this learning rate
    train_model(lr=lr, tracker=child_tracker)
    
    child_tracker.end_run()

tracker.end_run()
```

---

**For questions or issues, refer to the [MLflow documentation](https://mlflow.org/docs/latest/index.html)**
