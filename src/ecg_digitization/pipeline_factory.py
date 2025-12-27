"""
Pipeline factory for creating models based on configuration approach.

Enables A/B testing between baseline and SignalSavants methods.
"""

from typing import Dict, Any, Optional
import torch.nn as nn
import logging
from omegaconf import DictConfig

from ecg_digitization.data import HoughDeskewer
from ecg_digitization.models import get_segmenter, ECGEncoderDecoder, ECGDigitizer
from ecg_digitization.training import (
    SegmentationLoss,
    CombinedLoss,
    DiceLoss,
    FocalLoss,
)


class PipelineFactory:
    """Factory for creating pipeline components based on approach configuration."""
    
    def __init__(self, config: DictConfig):
        """Initialize factory with configuration.
        
        Args:
            config: Hydra configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.approach = config.approach.method
        self.logger.info(f"Initializing pipeline with approach: {self.approach}")
    
    def create_preprocessor(self) -> Optional[HoughDeskewer]:
        """Create preprocessor based on configuration.
        
        Returns:
            Preprocessor instance or None if disabled
        """
        if not self.config.approach.preprocessing.deskew:
            self.logger.info("Preprocessing: disabled (baseline mode)")
            return None
        
        method = self.config.approach.preprocessing.deskew_method
        self.logger.info(f"Preprocessing: Hough deskewing (method={method})")
        
        return HoughDeskewer(
            primary_method=method if method == "hough" else "gradient",
            fallback_method="gradient" if method == "hough" else "hough",
        )
    
    def create_segmenter(self, **kwargs) -> nn.Module:
        """Create segmentation model based on configuration.
        
        Args:
            **kwargs: Additional arguments for model creation
            
        Returns:
            Segmentation model
        """
        model_type = self.config.approach.segmentation.model
        use_fallback = self.config.approach.segmentation.use_fallback_on_error
        
        if model_type == "nnunet":
            self.logger.info("Segmentation: nnU-Net (SignalSavants)")
            try:
                return get_segmenter(
                    use_nnunet=True,
                    **kwargs
                )
            except Exception as e:
                if use_fallback:
                    self.logger.warning(f"nnU-Net failed ({e}), falling back to ECGDigitizer")
                    self._using_ecg_digitizer = True  # Track fallback
                    return ECGDigitizer(
                        encoder_name=self.config.model.encoder_name,
                        encoder_weights=self.config.model.encoder_weights,
                    )
                else:
                    raise
        
        elif model_type == "unet++":
            self.logger.info("Segmentation: UNet++ (baseline)")
            # Always use ECGDigitizer for training as trainer expects 'signals' output
            self.logger.info("Creating ECGDigitizer for training")
            self._using_ecg_digitizer = True
            return ECGDigitizer(
                encoder_name=self.config.model.encoder_name,
                encoder_weights=self.config.model.encoder_weights,
            )
        
        else:
            raise ValueError(f"Unknown segmentation model: {model_type}")
    
    def _create_unet_plus_plus(self, **kwargs) -> nn.Module:
        """Create UNet++ model.
        
        Args:
            **kwargs: Model arguments
            
        Returns:
            UNet++ model
        """
        encoder_name = kwargs.get("encoder_name", self.config.model.encoder_name)
        encoder_weights = kwargs.get("encoder_weights", self.config.model.encoder_weights)
        
        return ECGEncoderDecoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
        )
    
    def get_extraction_method(self) -> str:
        """Get signal extraction method.
        
        Returns:
            Extraction method: "column_wise" or "skeleton"
        """
        method = self.config.approach.extraction.method
        self.logger.info(f"Signal extraction: {method}")
        return method
    
    def get_vectorization_config(self) -> Dict[str, Any]:
        """Get vectorization configuration.
        
        Returns:
            Dictionary with vectorization parameters
        """
        return {
            "method": self.config.approach.extraction.vectorization.centroid_method,
            "interpolation": self.config.approach.extraction.vectorization.interpolation,
        }
    
    def create_loss(self) -> nn.Module:
        """Create loss function based on configuration.
        
        Returns:
            Loss function module
        """
        loss_type = self.config.approach.loss.type
        
        # If we're using ECGDigitizer (fallback), we need regression loss
        if getattr(self, '_using_ecg_digitizer', False) and loss_type == "segmentation":
            self.logger.warning("Using ECGDigitizer - switching to regression loss")
            loss_type = "regression"
        
        if loss_type == "segmentation":
            self.logger.info("Loss: Segmentation (Dice + CE) - SignalSavants")
            
            dice_weight = self.config.approach.loss.segmentation.dice_weight
            ce_weight = self.config.approach.loss.segmentation.ce_weight
            use_focal = self.config.approach.loss.segmentation.use_focal
            
            return SegmentationLoss(
                dice_weight=dice_weight,
                ce_weight=ce_weight,
                use_focal=use_focal,
            )
        
        elif loss_type == "regression":
            self.logger.info("Loss: Regression (SNR + MSE) - baseline")
            
            snr_weight = self.config.approach.loss.regression.snr_weight
            mse_weight = self.config.approach.loss.regression.mse_weight
            
            return CombinedLoss(
                snr_weight=snr_weight,
                mse_weight=mse_weight,
            )
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_calibration_config(self) -> Dict[str, Any]:
        """Get calibration configuration.
        
        Returns:
            Dictionary with calibration parameters
        """
        return {
            "use_calibration_pulse": self.config.approach.calibration.use_calibration_pulse,
            "use_grid_spacing": self.config.approach.calibration.use_grid_spacing,
            "use_blind_qrs": self.config.approach.calibration.use_blind_qrs,
            "fallback_chain": self.config.approach.calibration.fallback_chain,
        }
    
    def print_pipeline_summary(self):
        """Print a summary of the configured pipeline."""
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline Approach: {self.approach.upper()}")
        self.logger.info("=" * 60)
        self.logger.info(f"Preprocessing: {'Hough Deskewing' if self.config.approach.preprocessing.deskew else 'Basic (resize/normalize)'}")
        self.logger.info(f"Segmentation: {self.config.approach.segmentation.model.upper()}")
        self.logger.info(f"Extraction: {self.config.approach.extraction.method}")
        self.logger.info(f"Loss Function: {self.config.approach.loss.type}")
        self.logger.info(f"Calibration Chain: {' → '.join(self.config.approach.calibration.fallback_chain)}")
        self.logger.info("=" * 60)


def create_pipeline_from_config(config: DictConfig) -> PipelineFactory:
    """Convenience function to create pipeline factory.
    
    Args:
        config: Hydra configuration
        
    Returns:
        PipelineFactory instance
    """
    factory = PipelineFactory(config)
    factory.print_pipeline_summary()
    return factory
