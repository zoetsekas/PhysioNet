"""
nnU-Net integration for ECG semantic segmentation.

Wrapper around nnUNetv2 framework for automatic architecture configuration
and training pipeline.
"""

from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# Check if nnUNet is available
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    NNUNET_AVAILABLE = True
except ImportError:
    logger.warning("nnUNetv2 not installed. Using fallback UNet.")
    NNUNET_AVAILABLE = False


class nnUNetSegmenter(nn.Module):
    """
    nnU-Net wrapper for ECG segmentation.
    
    nnU-Net automatically configures:
    - Network depth based on image resolution
    - Patch size based on GPU memory
    - Batch size and learning rate
    """
    
    def __init__(
        self,
        fold: int = 0,
        checkpoint_dir: str = "models/nnunet",
        dataset_id: int = 100,
        configuration: str = "2d",
    ):
        """Initialize nnU-Net model.
        
        Args:
            fold: Cross-validation fold (0-4)
            checkpoint_dir: Directory for checkpoints
            dataset_id: nnU-Net dataset ID
            configuration: "2d" or "3d"
        """
        super().__init__()
        
        if not NNUNET_AVAILABLE:
            logger.error("nnUNetv2 not available. Cannot initialize.")
            raise ImportError("Please install: pip install nnunetv2")
        
        self.fold = fold
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset_id = dataset_id
        self.configuration = configuration
        
        # Will be initialized during training
        self.trainer = None
        self.predictor = None
    
    def setup_training(
        self,
        plans_file: Optional[str] = None,
        num_epochs: int = 1000,
    ):
        """Setup nnU-Net training.
        
        Args:
            plans_file: Path to plans file (auto-generated if None)
            num_epochs: Number of training epochs
        """
        self.trainer = nnUNetTrainer(
            plans=plans_file,
            fold=self.fold,
            dataset_json_path=self._get_dataset_json(),
            unpack_dataset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        self.trainer.initialize()
        logger.info(f"nnU-Net trainer initialized for fold {self.fold}")
    
    def setup_inference(self, checkpoint_path: str):
        """Setup nnU-Net for inference.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
        """
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        self.predictor.initialize_from_trained_model_folder(
            checkpoint_path,
            use_folds=(self.fold,),
            checkpoint_name="checkpoint_final.pth",
        )
        
        logger.info(f"nnU-Net predictor initialized from {checkpoint_path}")
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for compatibility.
        
        Note: nnU-Net uses its own inference pipeline.
        This is a wrapper for compatibility with our codebase.
        """
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Call setup_inference() first.")
        
        # Convert tensor to numpy
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # nnU-Net expects [batch, channel, height, width]
        # Our input is [batch, channel, height, width]
        predictions = []
        for img in x_np:
            pred = self.predictor.predict_single_npy_array(
                img,
                None,  # No preprocessing
                None,  # No output file
                save_or_return_probabilities=False,
            )
            predictions.append(pred)
        
        predictions = np.stack(predictions)
        return torch.from_numpy(predictions).to(x.device)
    
    def _get_dataset_json(self) -> str:
        """Get dataset JSON path for nnU-Net."""
        # This should point to nnU-Net dataset structure
        dataset_path = self.checkpoint_dir / f"Dataset{self.dataset_id:03d}_ECG"
        return str(dataset_path / "dataset.json")


class FallbackUNet(nn.Module):
    """Fallback U-Net when nnU-Net is not available."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.encoder.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._double_conv(feature * 2, feature))
        
        # Output
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx, (upconv, decoder_block) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            skip = skip_connections[idx]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)
        
        return self.output(x)


def get_segmenter(use_nnunet: bool = True, **kwargs) -> nn.Module:
    """Factory function to get appropriate segmenter.
    
    Args:
        use_nnunet: Try to use nnU-Net if available
        **kwargs: Arguments for segmenter
        
    Returns:
        Segmenter model
    """
    if use_nnunet and NNUNET_AVAILABLE:
        logger.info("Using nnU-Net segmenter")
        return nnUNetSegmenter(**kwargs)
    else:
        logger.info("Using fallback U-Net segmenter")
        return FallbackUNet(**kwargs)
