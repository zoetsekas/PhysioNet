"""
Backbone networks for feature extraction from ECG images.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import timm
import logging


def create_timm_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
    features_only: bool = True,
    out_indices: Optional[List[int]] = None,
    **kwargs,
) -> nn.Module:
    """Get a backbone network from timm.
    
    Args:
        name: Model name (see timm.list_models())
        pretrained: Whether to use pretrained weights
        features_only: Return intermediate features
        out_indices: Indices of feature levels to return
        **kwargs: Additional arguments for timm.create_model
        
    Returns:
        Backbone network
    """
    _logger = logging.getLogger(__name__)
    if out_indices is None:
        out_indices = [1, 2, 3, 4]  # Multi-scale features
    
    model = timm.create_model(
        name,
        pretrained=pretrained,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    
    _logger.info(f"Created backbone: {name} (pretrained={pretrained})")
    
    # Get feature info
    if hasattr(model, 'feature_info'):
        channels = model.feature_info.channels()
        _logger.info(f"Feature channels: {channels}")
    
    return model


class HuggingFaceBackbone(nn.Module):
    """Wrapper for HuggingFace vision models.
    
    Supports models like:
    - microsoft/swin-base-patch4-window7-224
    - google/vit-base-patch16-224
    - facebook/convnext-base-224
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/swin-base-patch4-window7-224",
        pretrained: bool = True,
        return_features: bool = True,
    ):
        """Initialize HuggingFace backbone.
        
        Args:
            model_name: HuggingFace model name
            pretrained: Whether to use pretrained weights
            return_features: Return intermediate features
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        from transformers import AutoModel, AutoConfig
        
        self.model_name = model_name
        self.return_features = return_features
        
        if pretrained:
            self.model = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=return_features,
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.output_hidden_states = return_features
            self.model = AutoModel.from_config(config)
        
        # Get output dimension
        config = self.model.config
        if hasattr(config, 'hidden_size'):
            self.hidden_size = config.hidden_size
        elif hasattr(config, 'hidden_sizes'):
            self.hidden_size = config.hidden_sizes[-1]
        else:
            self.hidden_size = 768  # Default
            
        self.logger.info(f"Created HuggingFace backbone: {model_name} (hidden_size={self.hidden_size})")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Either final features or list of hidden states
        """
        outputs = self.model(x)
        
        if self.return_features:
            return outputs.hidden_states
        else:
            return outputs.last_hidden_state


class MultiScaleFeatureExtractor(nn.Module):
    """Extract multi-scale features from backbone.
    
    Useful for combining features from different scales for
    better signal extraction at various resolutions.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: List[int],
        out_channels: int = 256,
    ):
        """Initialize multi-scale feature extractor.
        
        Args:
            backbone: Backbone network
            feature_channels: Channel dimensions from backbone
            out_channels: Output channel dimension
        """
        super().__init__()
        self.backbone = backbone
        
        # Projection layers to unify channel dimensions
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1)
            for ch in feature_channels
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(feature_channels), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Fused multi-scale features [B, out_channels, H/4, W/4]
        """
        # Get backbone features
        features = self.backbone(x)
        
        # Project and resize to common resolution
        target_size = features[0].shape[2:]  # Use first feature map size
        
        projected = []
        for feat, proj in zip(features, self.projections):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = nn.functional.interpolate(
                    p, size=target_size, mode='bilinear', align_corners=False
                )
            projected.append(p)
        
        # Concatenate and fuse
        concat = torch.cat(projected, dim=1)
        fused = self.fusion(concat)
        
        return fused
