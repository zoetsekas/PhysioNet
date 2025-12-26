"""
ECG Digitization Trainer.
"""

from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger


class ECGTrainer:
    """Training loop for ECG digitization."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "models/check points",
        mlflow_tracker: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.mlflow_tracker = mlflow_tracker
        
        # Track losses for visualization
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch["images"].to(self.device)
            signals = batch["signals"].to(self.device)
            masks = batch.get("signal_masks", None)
            if masks is not None:
                masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, target_length=signals.shape[-1])
            loss = self.criterion(outputs["signals"], signals, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        # Log to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metric("train_loss", avg_loss, step=epoch)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["images"].to(self.device)
            signals = batch["signals"].to(self.device)
            
            outputs = self.model(images, target_length=signals.shape[-1])
            loss = self.criterion(outputs["signals"], signals)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Log to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metric("val_loss", avg_loss, step=epoch)
            
            # Calculate and log SNR (negative loss is approximation)
            val_snr = -avg_loss
            self.mlflow_tracker.log_metric("val_snr", val_snr, step=epoch)
        
        return avg_loss
    
    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                
                # Log best model to MLflow
                if self.mlflow_tracker:
                    self.mlflow_tracker.log_metric("best_val_loss", val_loss, step=epoch)
        
        # Log final visualizations to MLflow
        if self.mlflow_tracker:
            self._log_training_visualizations()
            self._log_final_model()
    
    def _log_training_visualizations(self):
        """Log training visualizations to MLflow."""
        from ecg_digitization.utils.mlflow_utils import create_loss_plot
        
        # Loss curves
        loss_fig = create_loss_plot(self.train_losses, self.val_losses)
        self.mlflow_tracker.log_figure(loss_fig, "loss_curves.png")
        
        logger.info("Logged training visualizations to MLflow")
    
    def _log_final_model(self):
        """Log final model to MLflow."""
        try:
            model_path = self.checkpoint_dir / "best_model.pt"
            if model_path.exists():
                self.mlflow_tracker.log_artifact(str(model_path), "models")
                logger.info("Logged best model to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")
