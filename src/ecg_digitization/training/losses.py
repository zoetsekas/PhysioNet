"""
Loss functions for ECG digitization training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNRLoss(nn.Module):
    """Signal-to-Noise Ratio loss (competition metric)."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute negative SNR (for minimization).
        
        Args:
            pred: Predicted signals [B, leads, T]
            target: Ground truth signals [B, leads, T]
            mask: Optional mask for valid samples
            
        Returns:
            Negative mean SNR in dB
        """
        error = target - pred
        
        signal_power = (target ** 2).sum(dim=-1)
        noise_power = (error ** 2).sum(dim=-1)
        
        snr = 10 * torch.log10(signal_power / (noise_power + self.eps) + self.eps)
        
        if mask is not None:
            snr = snr * mask
            return -snr.sum() / (mask.sum() + self.eps)
        
        return -snr.mean()


class CombinedLoss(nn.Module):
    """Combined loss with MSE and SNR components."""
    
    def __init__(self, snr_weight: float = 1.0, mse_weight: float = 0.1):
        super().__init__()
        self.snr_loss = SNRLoss()
        self.mse_weight = mse_weight
        self.snr_weight = snr_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        snr = self.snr_loss(pred, target, mask)
        mse = F.mse_loss(pred, target)
        return self.snr_weight * snr + self.mse_weight * mse
