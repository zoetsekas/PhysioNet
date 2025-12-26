"""
ECG Image Dataset for training and inference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


class ECGImageDataset(Dataset):
    """Dataset for ECG image digitization.
    
    Loads ECG images and their corresponding time-series ground truth signals.
    
    Attributes:
        data_dir: Root directory containing the data
        metadata: DataFrame with image metadata
        transform: Optional image transforms to apply
        is_train: Whether this is training data (has ground truth)
        leads: List of ECG lead names
    """
    
    # Standard 12-lead ECG order
    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        transform: Optional[callable] = None,
        is_train: bool = True,
        image_suffix: str = "-0001.png",  # Default to original scan
        max_samples: Optional[int] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Root directory containing train/test folders
            metadata_path: Path to train.csv or test.csv
            transform: Optional albumentations transform
            is_train: Whether this is training data
            image_suffix: Image file suffix to load
            max_samples: Optional limit on number of samples
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.image_suffix = image_suffix
        
        # Load metadata
        if metadata_path is None:
            metadata_path = self.data_dir / ("train.csv" if is_train else "test.csv")
        self.metadata = pd.read_csv(metadata_path)
        
        # Get unique record IDs
        self.record_ids = self.metadata["id"].unique().tolist()
        
        if max_samples is not None:
            self.record_ids = self.record_ids[:max_samples]
            
        logger.info(f"Loaded {len(self.record_ids)} records from {metadata_path}")
        
    def __len__(self) -> int:
        return len(self.record_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an ECG image and its ground truth signal.
        
        Returns:
            Dictionary containing:
                - image: Preprocessed ECG image tensor [C, H, W]
                - signal: Ground truth signal tensor [12, T] (if training)
                - record_id: Record identifier
                - metadata: Additional metadata dict
        """
        record_id = self.record_ids[idx]
        
        # Load image
        image = self._load_image(record_id)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Convert to tensor [C, H, W]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        result = {
            "image": image,
            "record_id": record_id,
        }
        
        # Load ground truth if training
        if self.is_train:
            signal = self._load_signal(record_id)
            result["signal"] = torch.from_numpy(signal).float()
            
        # Add metadata
        record_meta = self.metadata[self.metadata["id"] == record_id].iloc[0]
        result["metadata"] = {
            "fs": record_meta.get("fs", 500),  # Sampling frequency
            "number_of_rows": record_meta.get("number_of_rows", 4),
        }
        
        return result
    
    def _load_image(self, record_id: str) -> np.ndarray:
        """Load ECG image for a record.
        
        Args:
            record_id: Record identifier
            
        Returns:
            Image as numpy array [H, W, C] in RGB
        """
        subdir = "train" if self.is_train else "test"
        image_path = self.data_dir / subdir / record_id / f"{record_id}{self.image_suffix}"
        
        if not image_path.exists():
            # Try alternative image formats
            for suffix in ["-0001.png", "-0002.png", "-0001.jpg", ".png", ".jpg"]:
                alt_path = self.data_dir / subdir / record_id / f"{record_id}{suffix}"
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_signal(self, record_id: str) -> np.ndarray:
        """Load ground truth ECG signal for a record.
        
        Args:
            record_id: Record identifier
            
        Returns:
            Signal as numpy array [12, T] where T is number of time samples
        """
        signal_path = self.data_dir / "train" / record_id / f"{record_id}.csv"
        
        if not signal_path.exists():
            raise FileNotFoundError(f"Could not load signal: {signal_path}")
        
        # Load signal CSV - columns are lead names, rows are time samples
        signal_df = pd.read_csv(signal_path)
        
        # Ensure correct lead order
        signals = []
        for lead in self.LEADS:
            if lead in signal_df.columns:
                signals.append(signal_df[lead].values)
            else:
                logger.warning(f"Lead {lead} not found in {signal_path}")
                signals.append(np.zeros(len(signal_df)))
                
        return np.stack(signals, axis=0)  # [12, T]
    
    def get_sample_by_id(self, record_id: str) -> Dict[str, torch.Tensor]:
        """Get sample by record ID instead of index."""
        idx = self.record_ids.index(record_id)
        return self[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable-length signals.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with padded sequences
    """
    images = torch.stack([sample["image"] for sample in batch])
    record_ids = [sample["record_id"] for sample in batch]
    
    result = {
        "images": images,
        "record_ids": record_ids,
    }
    
    # Handle signals if present (training)
    if "signal" in batch[0]:
        # Pad signals to same length
        max_len = max(sample["signal"].shape[1] for sample in batch)
        padded_signals = []
        signal_masks = []
        
        for sample in batch:
            signal = sample["signal"]
            pad_len = max_len - signal.shape[1]
            if pad_len > 0:
                signal = torch.nn.functional.pad(signal, (0, pad_len))
                mask = torch.cat([
                    torch.ones(sample["signal"].shape[1]),
                    torch.zeros(pad_len)
                ])
            else:
                mask = torch.ones(signal.shape[1])
            
            padded_signals.append(signal)
            signal_masks.append(mask)
        
        result["signals"] = torch.stack(padded_signals)
        result["signal_masks"] = torch.stack(signal_masks)
    
    return result
