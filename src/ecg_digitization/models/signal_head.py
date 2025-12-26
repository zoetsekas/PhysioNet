"""
Signal regression head for converting image features to ECG signals.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class SignalRegressionHead(nn.Module):
    """Regresses ECG signals from 2D image features."""
    
    def __init__(
        self,
        in_channels: int = 64,
        num_leads: int = 12,
        signal_length: int = 5000,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.signal_length = signal_length
        
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, 1)
        
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.decoder = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(hidden_dim, num_leads, 1)
        
        logger.info(f"Created SignalRegressionHead: {num_leads} leads")
    
    def forward(self, features: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        if target_length is None:
            target_length = self.signal_length
        
        x = self.height_pool(features).squeeze(2)
        x = self.input_proj(x)
        x = self.decoder(x)
        x = self.output_proj(x)
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        return x
