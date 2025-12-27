"""
Ray-based training with distributed data loading and hyperparameter tuning.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging


def train_func(config: Dict[str, Any]):
    """Training function for Ray Train.
    
    This function runs on each worker and handles the training loop.
    """
    from ecg_digitization.data import ECGImageDataset, get_train_transforms, get_val_transforms, collate_fn
    from ecg_digitization.models import ECGDigitizer
    from ecg_digitization.training import CombinedLoss
    
    import ray.train as train
    from ray.train import Checkpoint
    
    # Get data directory from config
    data_dir = config.get("data_dir", "data")
    
    # Create transforms
    image_size = (config.get("image_height", 1024), config.get("image_width", 1280))
    train_transform = get_train_transforms(image_size, config.get("augment_prob", 0.5))
    val_transform = get_val_transforms(image_size)
    
    # Create dataset
    full_dataset = ECGImageDataset(
        data_dir,
        transform=train_transform,
        is_train=True,
    )
    
    # Split dataset
    val_size = int(len(full_dataset) * config.get("val_split", 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Create model
    model = ECGDigitizer(
        encoder_name=config.get("encoder_name", "resnet50"),
        encoder_weights=config.get("encoder_weights", "imagenet"),
        num_leads=config.get("num_leads", 12),
        signal_length=config.get("signal_length", 5000),
        hidden_dim=config.get("hidden_dim", 256),
    )
    
    # Prepare model for distributed training
    model = train.torch.prepare_model(model)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 50)
    )
    
    # Loss function
    criterion = CombinedLoss(
        snr_weight=config.get("snr_weight", 1.0),
        mse_weight=config.get("mse_weight", 0.1),
    )
    
    # Training loop
    for epoch in range(config.get("epochs", 50)):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch["images"].cuda()
            signals = batch["signals"].cuda()
            masks = batch.get("signal_masks")
            if masks is not None:
                masks = masks.cuda()
            
            optimizer.zero_grad()
            outputs = model(images, target_length=signals.shape[-1])
            loss = criterion(outputs["signals"], signals, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].cuda()
                signals = batch["signals"].cuda()
                outputs = model(images, target_length=signals.shape[-1])
                loss = criterion(outputs["signals"], signals)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        # Calculate SNR (negative loss is SNR)
        val_snr = -val_loss
        
        # Report metrics to Ray
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_snr": val_snr,
            "lr": scheduler.get_last_lr()[0],
        }
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                os.path.join(temp_dir, "model.pt")
            )
            train.report(metrics, checkpoint=Checkpoint.from_directory(temp_dir))

import tempfile


def tune_trainable(config: Dict[str, Any]):
    """Trainable function for Ray Tune (single GPU per trial).
    
    This is a simplified version that doesn't use Ray Train's distributed APIs.
    """
    from ecg_digitization.data import ECGImageDataset, get_train_transforms, get_val_transforms, collate_fn
    from ecg_digitization.models import ECGDigitizer
    from ecg_digitization.training import CombinedLoss
    from ray import tune
    
    # Get data directory from config
    data_dir = config.get("data_dir", "/app/data")
    
    # Create transforms
    image_size = (config.get("image_height", 512), config.get("image_width", 640))
    train_transform = get_train_transforms(image_size, config.get("augment_prob", 0.5))
    val_transform = get_val_transforms(image_size)
    
    # Create dataset
    full_dataset = ECGImageDataset(
        data_dir,
        transform=train_transform,
        is_train=True,
    )
    
    # Split dataset
    val_size = int(len(full_dataset) * config.get("val_split", 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Create model
    model = ECGDigitizer(
        encoder_name=config.get("encoder_name", "resnet50"),
        encoder_weights=config.get("encoder_weights", "imagenet"),
        num_leads=config.get("num_leads", 12),
        signal_length=config.get("signal_length", 5000),
        hidden_dim=config.get("hidden_dim", 256),
    )
    model = model.cuda()
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    
    epochs = config.get("epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function
    criterion = CombinedLoss(
        snr_weight=config.get("snr_weight", 1.0),
        mse_weight=config.get("mse_weight", 0.1),
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch["images"].cuda()
            signals = batch["signals"].cuda()
            masks = batch.get("signal_masks")
            if masks is not None:
                masks = masks.cuda()
            
            optimizer.zero_grad()
            outputs = model(images, target_length=signals.shape[-1])
            loss = criterion(outputs["signals"], signals, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].cuda()
                signals = batch["signals"].cuda()
                outputs = model(images, target_length=signals.shape[-1])
                loss = criterion(outputs["signals"], signals)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        # Calculate SNR (negative loss is SNR)
        val_snr = -val_loss
        
        # Report metrics to Ray Tune
        tune.report({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_snr": val_snr,
            "training_iteration": epoch + 1,
        })


class RayTrainer:
    """Ray-based trainer with distributed training and hyperparameter tuning."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        num_workers: int = 1,
        use_gpu: bool = True,
        resources_per_worker: Optional[Dict[str, float]] = None,
    ):
        """Initialize Ray trainer.
        
        Args:
            config: Training configuration
            num_workers: Number of distributed workers
            use_gpu: Whether to use GPU
            resources_per_worker: Resource allocation per worker
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        
        if resources_per_worker is None:
            resources_per_worker = {"CPU": 4, "GPU": 1}
        self.resources_per_worker = resources_per_worker
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def train(self) -> ray.train.Result:
        """Run training with Ray Train."""
        
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker=self.resources_per_worker,
        )
        
        run_config = RunConfig(
            name="ecg_digitization_train",
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_snr",
                checkpoint_score_order="max",
            ),
        )
        
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=self.config,
            scaling_config=scaling_config,
            run_config=run_config,
        )
        
        result = trainer.fit()
        self.logger.info(f"Training completed. Best SNR: {result.metrics.get('val_snr', 'N/A')}")
        return result
    
    def tune(
        self,
        num_samples: int = 20,
        max_epochs: int = 50,
        grace_period: int = 5,
    ) -> ray.tune.ResultGrid:
        """Run hyperparameter tuning with Ray Tune.
        
        Args:
            num_samples: Number of hyperparameter configurations to try
            max_epochs: Maximum epochs per trial
            grace_period: Minimum epochs before early stopping
            
        Returns:
            ResultGrid with tuning results
        """
        # Define search space - merge with base config
        search_space = {
            **self.config,
            "epochs": max_epochs,  # Set epochs for tuning
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([2, 4]),
            "hidden_dim": tune.choice([128, 256]),
            "snr_weight": tune.uniform(0.5, 2.0),
            "mse_weight": tune.uniform(0.05, 0.2),
            "augment_prob": tune.uniform(0.3, 0.7),
            "weight_decay": tune.loguniform(1e-6, 1e-4),
        }
        
        # ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_epochs,
            grace_period=grace_period,
            reduction_factor=2,
        )
        
        # Create tuner following the example pattern
        tuner = tune.Tuner(
            tune.with_resources(
                tune_trainable,
                resources={"cpu": 4, "gpu": 1}
            ),
            tune_config=tune.TuneConfig(
                metric="val_snr",
                mode="max",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=search_space,
        )
        
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result("val_snr", "max")
        
        self.logger.info(f"Best trial config: {best_result.config}")
        self.logger.info(f"Best trial final val_snr: {best_result.metrics['val_snr']}")
        self.logger.info(f"Best trial final val_loss: {best_result.metrics['val_loss']}")
        
        return results

