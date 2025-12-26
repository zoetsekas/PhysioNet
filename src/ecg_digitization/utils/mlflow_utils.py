"""
MLflow integration for experiment tracking, model logging, and visualization.

Provides comprehensive tracking of ECG digitization experiments.
Can be disabled for environments without MLflow (e.g., Kaggle).
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn

# Try to import MLflow, but make it optional
try:
    import mlflow
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.getLogger(__name__).warning("MLflow not available - tracking will be disabled")


class NoOpMLflowTracker:
    """No-op MLflow tracker for when MLflow is disabled or unavailable."""
    
    def __init__(self, *args, **kwargs):
        """Initialize no-op tracker."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("MLflow tracking disabled - using no-op tracker")
        self.run = None
    
    def start_run(self, *args, **kwargs):
        """No-op start run."""
        return None
    
    def log_config(self, config: Dict[str, Any]):
        """No-op log config."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """No-op log metrics."""
        pass
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """No-op log metric."""
        pass
    
    def log_model(self, *args, **kwargs):
        """No-op log model."""
        pass
    
    def log_artifact(self, *args, **kwargs):
        """No-op log artifact."""
        pass
    
    def log_artifacts(self, *args, **kwargs):
        """No-op log artifacts."""
        pass
    
    def log_figure(self, *args, **kwargs):
        """No-op log figure."""
        pass
    
    def log_dict(self, *args, **kwargs):
        """No-op log dict."""
        pass
    
    def set_tags(self, *args, **kwargs):
        """No-op set tags."""
        pass
    
    def end_run(self, *args, **kwargs):
        """No-op end run."""
        pass


def create_mlflow_tracker(
    enabled: bool = True,
    tracking_uri: str = "http://localhost:5050",
    experiment_name: str = "ecg-digitization",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """Factory function to create appropriate MLflow tracker.
    
    Args:
        enabled: Whether MLflow tracking is enabled
        tracking_uri: MLflow server URI
        experiment_name: Name of the experiment
        run_name: Optional run name
        tags: Optional tags for the run
        
    Returns:
        MLflowExperimentTracker if enabled and available, else NoOpMLflowTracker
    """
    _logger = logging.getLogger(__name__)
    if not enabled:
        _logger.info("MLflow tracking disabled by configuration")
        return NoOpMLflowTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
        )
    
    if not MLFLOW_AVAILABLE:
        _logger.warning("MLflow not available - falling back to no-op tracker")
        return NoOpMLflowTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
        )
    
    _logger.info(f"MLflow tracking enabled: {tracking_uri}")
    return MLflowExperimentTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
    )


class MLflowExperimentTracker:
    """Comprehensive MLflow experiment tracker for ECG digitization."""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5050",
        experiment_name: str = "ecg-digitization",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow server URI
            experiment_name: Name of the experiment
            run_name: Optional run name
            tags: Optional tags for the run
        """
        self.logger = logging.getLogger(__name__)
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags={"project": "physionet-ecg", "competition": "2024"}
                )
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name=experiment_name)
            self.logger.info(f"MLflow experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            self.logger.warning(f"Failed to set experiment: {e}")
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.run = None
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Start MLflow run.
        
        Args:
            run_name: Optional run name override
            nested: Whether this is a nested run
        """
        run_name = run_name or self.run_name
        
        self.run = mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=self.tags,
        )
        
        self.logger.info(f"Started MLflow run: {self.run.info.run_id}")
        return self.run
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters.
        
        Args:
            config: Configuration dictionary
        """
        # Flatten nested config
        flat_params = self._flatten_dict(config)
        
        # Log to MLflow
        mlflow.log_params(flat_params)
        self.logger.info(f"Logged {len(flat_params)} configuration parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
    ):
        """Log PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path within artifact storage
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
        """
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        self.logger.info(f"Logged model to {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file.
        
        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifact storage
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log a directory of artifacts.
        
        Args:
            local_dir: Path to local directory
            artifact_path: Optional path within artifact storage
        """
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_figure(
        self,
        figure: plt.Figure,
        artifact_file: str,
    ):
        """Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            artifact_file: Filename for the artifact
        """
        # Save figure temporarily
        temp_path = Path("/tmp") / artifact_file
        figure.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close(figure)
        
        # Log to MLflow
        mlflow.log_artifact(str(temp_path))
        
        # Clean up
        temp_path.unlink()
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log a dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact
        """
        mlflow.log_dict(dictionary, artifact_file)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run.
        
        Args:
            tags: Dictionary of tag name -> value
        """
        mlflow.set_tags(tags)
    
    def end_run(self, status: str = "FINISHED"):
        """End the current run.
        
        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED")
        """
        if self.run:
            mlflow.end_run(status=status)
            self.logger.info(f"Ended MLflow run with status: {status}")
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.',
    ) -> Dict[str, Any]:
        """Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert to string for lists/tuples
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)


def create_loss_plot(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training and Validation Loss",
) -> plt.Figure:
    """Create loss curve plot.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return fig


def create_snr_plot(
    snr_scores: Dict[str, List[float]],
    title: str = "SNR per Lead",
) -> plt.Figure:
    """Create SNR bar plot.
    
    Args:
        snr_scores: Dictionary of lead name -> SNR values
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    leads = list(snr_scores.keys())
    snrs = [np.mean(scores) for scores in snr_scores.values()]
    stds = [np.std(scores) for scores in snr_scores.values()]
    
    bars = ax.bar(leads, snrs, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    
    # Color code by performance
    for i, (bar, snr) in enumerate(zip(bars, snrs)):
        if snr > 20:
            bar.set_color('green')
        elif snr > 15:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Lead', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Target (20 dB)')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    return fig


def create_signal_comparison_plot(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    lead_name: str = "Lead II",
    fs: int = 500,
) -> plt.Figure:
    """Create signal comparison plot.
    
    Args:
        ground_truth: Ground truth signal
        prediction: Predicted signal
        lead_name: Name of the lead
        fs: Sampling frequency
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    time = np.arange(len(ground_truth)) / fs
    
    # Ground truth
    ax1.plot(time, ground_truth, 'b-', linewidth=1.5, label='Ground Truth')
    ax1.set_ylabel('Amplitude (mV)', fontsize=11)
    ax1.set_title(f'{lead_name} - Ground Truth', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Prediction
    ax2.plot(time, prediction, 'r-', linewidth=1.5, label='Prediction')
    ax2.set_ylabel('Amplitude (mV)', fontsize=11)
    ax2.set_title(f'{lead_name} - Prediction', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Overlay
    ax3.plot(time, ground_truth, 'b-', linewidth=1.5, alpha=0.7, label='Ground Truth')
    ax3.plot(time, prediction, 'r--', linewidth=1.5, alpha=0.7, label='Prediction')
    error = np.abs(ground_truth - prediction)
    ax3.fill_between(time, 0, error, alpha=0.3, color='gray', label='Error')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Amplitude (mV)', fontsize=11)
    ax3.set_title(f'{lead_name} - Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    return fig
