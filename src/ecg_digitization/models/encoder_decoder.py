"""
Encoder-Decoder architecture for ECG image understanding.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from loguru import logger


class ECGEncoderDecoder(nn.Module):
    """Encoder-Decoder model for ECG image feature extraction.
    
    Uses segmentation-models-pytorch for flexible encoder-decoder
    architectures with pretrained encoders.
    
    The model outputs:
    1. Feature maps for signal extraction
    2. Optional segmentation mask for lead detection
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        output_stride: int = 4,
        decoder_attention_type: Optional[str] = "scse",
    ):
        """Initialize encoder-decoder.
        
        Args:
            encoder_name: Name of encoder backbone
            encoder_weights: Pretrained weights to use
            in_channels: Input image channels
            decoder_channels: Decoder feature channels
            output_stride: Output stride (downsampling factor)
            decoder_attention_type: Type of attention in decoder
        """
        super().__init__()
        
        # Use UNet++ for good multi-scale feature fusion
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,  # Placeholder, we use custom head
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
        )
        
        # Get encoder output channels
        self.encoder_channels = self.model.encoder.out_channels
        self.decoder_channels = decoder_channels
        
        # Feature output head (instead of segmentation)
        self.feature_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Optional lead segmentation head
        self.segment_leads = nn.Conv2d(64, 13, 1)  # 12 leads + background
        
        logger.info(f"Created ECGEncoderDecoder with encoder={encoder_name}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_segmentation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input image [B, C, H, W]
            return_segmentation: Whether to return lead segmentation
            
        Returns:
            Dictionary with:
                - features: Dense feature map [B, 64, H/4, W/4]
                - encoder_features: List of encoder features
                - segmentation: Optional lead segmentation [B, 13, H/4, W/4]
        """
        # Encoder features
        encoder_features = self.model.encoder(x)
        
        # Decoder
        decoder_out = self.model.decoder(*encoder_features)
        
        # Feature head
        features = self.feature_head(decoder_out)
        
        outputs = {
            "features": features,
            "encoder_features": encoder_features,
        }
        
        if return_segmentation:
            outputs["segmentation"] = self.segment_leads(features)
        
        return outputs


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return self.relu(out)


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale features."""
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        
        self.out_channels = out_channels
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.
        
        Args:
            features: List of feature maps from backbone (low to high resolution)
            
        Returns:
            List of FPN feature maps
        """
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )
        
        # Output
        outputs = [o(l) for o, l in zip(self.output_convs, laterals)]
        
        return outputs
