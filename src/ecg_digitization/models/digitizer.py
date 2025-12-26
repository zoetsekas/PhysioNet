"""
Main ECG Digitizer model combining all components.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import logging

from .encoder_decoder import ECGEncoderDecoder
from .signal_head import SignalRegressionHead


class ECGDigitizer(nn.Module):
    """End-to-end ECG digitization model.
    
    Takes ECG images and outputs time-series signals for all 12 leads.
    """
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        num_leads: int = 12,
        signal_length: int = 5000,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.num_leads = num_leads
        self.signal_length = signal_length
        
        # Encoder-decoder for feature extraction
        self.encoder_decoder = ECGEncoderDecoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
        )
        
        # Signal regression head
        self.signal_head = SignalRegressionHead(
            in_channels=64,
            num_leads=num_leads,
            signal_length=signal_length,
            hidden_dim=hidden_dim,
        )
        
        self.logger.info(f"Created ECGDigitizer: {encoder_name}, {num_leads} leads")
    
    def forward(
        self,
        images: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            target_length: Optional target signal length
            
        Returns:
            Dictionary with 'signals' [B, num_leads, T]
        """
        # Extract features
        feat_dict = self.encoder_decoder(images)
        features = feat_dict["features"]
        
        # Regress signals
        signals = self.signal_head(features, target_length)
        
        return {"signals": signals, "features": features}
