"""
Ray-based parallel data preprocessing pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import ray
from ray import data as ray_data
import numpy as np
import pandas as pd
from loguru import logger


@ray.remote
class DataPreprocessor:
    """Ray actor for parallel data preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        import cv2
        self.config = config
        self.image_size = (config.get("image_height", 1024), config.get("image_width", 1280))
    
    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocess a single ECG image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with preprocessed data
        """
        import cv2
        from ecg_digitization.data.preprocessing import preprocess_ecg_image, extract_grid_info
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Failed to load {image_path}"}
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract grid info for calibration
        grid_info = extract_grid_info(image)
        
        # Preprocess
        processed = preprocess_ecg_image(image)
        
        # Resize
        processed = cv2.resize(processed, (self.image_size[1], self.image_size[0]))
        
        return {
            "path": image_path,
            "image": processed,
            "grid_info": grid_info,
        }


def create_ray_dataset(
    data_dir: str,
    is_train: bool = True,
    num_cpus: int = 4,
) -> ray_data.Dataset:
    """Create a Ray Dataset for parallel data loading.
    
    Args:
        data_dir: Data directory path
        is_train: Whether this is training data
        num_cpus: Number of CPUs for parallelism
        
    Returns:
        Ray Dataset
    """
    data_dir = Path(data_dir)
    subdir = "train" if is_train else "test"
    
    # Get all image paths
    image_paths = []
    for record_dir in (data_dir / subdir).iterdir():
        if record_dir.is_dir():
            for img_file in record_dir.glob("*.png"):
                image_paths.append(str(img_file))
            for img_file in record_dir.glob("*.jpg"):
                image_paths.append(str(img_file))
    
    logger.info(f"Found {len(image_paths)} images in {data_dir / subdir}")
    
    # Create Ray Dataset
    ds = ray_data.from_items([{"path": p} for p in image_paths])
    
    return ds


def preprocess_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Batch preprocessing function for Ray Data.
    
    Args:
        batch: Batch of image paths
        
    Returns:
        Preprocessed batch
    """
    import cv2
    from ecg_digitization.data.preprocessing import preprocess_ecg_image
    
    processed_images = []
    for path in batch["path"]:
        image = cv2.imread(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_ecg_image(image)
            image = cv2.resize(image, (1280, 1024))
            processed_images.append(image)
        else:
            processed_images.append(np.zeros((1024, 1280, 3), dtype=np.uint8))
    
    batch["image"] = np.stack(processed_images)
    return batch


@ray.remote(num_gpus=1)
class GPUPredictor:
    """Ray actor for GPU-based inference."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """Initialize GPU predictor.
        
        Args:
            model_path: Path to model checkpoint
            config: Model configuration
        """
        import torch
        from ecg_digitization.models import ECGDigitizer
        
        self.device = "cuda"
        
        self.model = ECGDigitizer(
            encoder_name=config.get("encoder_name", "resnet50"),
            num_leads=config.get("num_leads", 12),
            signal_length=config.get("signal_length", 5000),
        )
        
        # Load checkpoint
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()
        
        self.config = config
    
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Run inference on a batch of images.
        
        Args:
            images: Batch of images [B, H, W, C]
            
        Returns:
            Predicted signals [B, 12, T]
        """
        import torch
        
        # Convert to tensor and normalize
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            signals = outputs["signals"].cpu().numpy()
        
        return signals


class ParallelInferencePipeline:
    """Ray-based parallel inference pipeline."""
    
    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any],
        num_actors: int = 1,
    ):
        """Initialize parallel inference pipeline.
        
        Args:
            model_path: Path to model checkpoint
            config: Model configuration
            num_actors: Number of GPU actors
        """
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        self.predictors = [
            GPUPredictor.remote(model_path, config)
            for _ in range(num_actors)
        ]
        self.config = config
        self.current_actor = 0
    
    def predict_dataset(
        self,
        dataset: ray_data.Dataset,
        batch_size: int = 8,
    ) -> Dict[str, np.ndarray]:
        """Run inference on entire dataset.
        
        Args:
            dataset: Ray Dataset with images
            batch_size: Batch size for inference
            
        Returns:
            Dictionary mapping record IDs to predictions
        """
        # Preprocess images in parallel
        preprocessed = dataset.map_batches(
            preprocess_batch,
            batch_size=batch_size,
            num_cpus=4,
        )
        
        # Collect predictions
        predictions = {}
        
        for batch in preprocessed.iter_batches(batch_size=batch_size):
            # Round-robin across actors
            actor = self.predictors[self.current_actor]
            self.current_actor = (self.current_actor + 1) % len(self.predictors)
            
            # Get predictions
            signals = ray.get(actor.predict_batch.remote(batch["image"]))
            
            for i, path in enumerate(batch["path"]):
                record_id = Path(path).parent.name
                predictions[record_id] = signals[i]
        
        return predictions
