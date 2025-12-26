"""
ECG Predictor for inference and submission generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging


class ECGPredictor:
    """Inference predictor for ECG digitization."""
    
    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model = model.to(device)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
    
    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.logger.info(f"Loaded checkpoint: {path}")
    
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        signal_length: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """Run inference on a dataloader.
        
        Returns:
            Dict mapping record_id to predictions [12, T]
        """
        predictions = {}
        
        for batch in tqdm(dataloader, desc="Predicting"):
            images = batch["images"].to(self.device)
            record_ids = batch["record_ids"]
            
            outputs = self.model(images, target_length=signal_length)
            signals = outputs["signals"].cpu().numpy()
            
            for i, rid in enumerate(record_ids):
                predictions[rid] = signals[i]
        
        return predictions
    
    def generate_submission(
        self,
        predictions: Dict[str, np.ndarray],
        output_path: str,
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate submission file in competition format.
        
        Format: id,value where id = {record_id}_{row_id}_{lead}
        """
        rows = []
        
        for record_id, signals in predictions.items():
            record_meta = metadata[metadata["id"] == record_id]
            fs = record_meta["fs"].iloc[0] if len(record_meta) > 0 else 500
            
            for lead_idx, lead in enumerate(self.LEADS):
                signal = signals[lead_idx]
                for row_id, value in enumerate(signal):
                    rows.append({
                        "id": f"{record_id}_{row_id}_{lead}",
                        "value": float(value),
                    })
        
        df = pd.DataFrame(rows)
        
        # Save as parquet (preferred) or CSV
        if output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Saved submission to {output_path}")
        return df
