"""Model architectures for ECG digitization."""

from .backbone import create_timm_backbone, HuggingFaceBackbone
from .encoder_decoder import ECGEncoderDecoder
from .signal_head import SignalRegressionHead
from .nnunet_segmenter import nnUNetSegmenter, FallbackUNet, get_segmenter
from .digitizer import ECGDigitizer

__all__ = [
    "create_timm_backbone",
    "HuggingFaceBackbone",
    "ECGEncoderDecoder",
    "SignalRegressionHead",
    "nnUNetSegmenter",
    "FallbackUNet",
    "get_segmenter",
    "ECGDigitizer",
]
