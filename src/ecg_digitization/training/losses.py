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
            mask: Optional mask for valid samples [B, T]
            
        Returns:
            Negative mean SNR in dB
        """
        # Handle NaNs in target (from empty columns in CSV)
        valid_mask = ~torch.isnan(target)
        target = torch.where(valid_mask, target, torch.zeros_like(target))
        pred = torch.where(valid_mask, pred, torch.zeros_like(pred))
        
        # Combine with temporal mask if provided
        if mask is not None:
            # mask is [B, T], make it [B, 1, T] for broadcasting
            valid_mask = valid_mask & mask.unsqueeze(1).bool()
        
        error = target - pred
        
        # Apply mask to elements
        target = target * valid_mask
        error = error * valid_mask
        
        # Compute power per lead
        signal_power = (target ** 2).sum(dim=-1)
        noise_power = (error ** 2).sum(dim=-1)
        
        # Compute SNR in dB
        # Add eps inside log and in divisor to avoid nan/inf
        snr_ratio = signal_power / (noise_power + self.eps)
        snr = 10 * torch.log10(snr_ratio + self.eps)
        
        # Average only over valid leads
        lead_has_data = valid_mask.any(dim=-1)
        if lead_has_data.any():
            return -snr[lead_has_data].mean()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class CombinedLoss(nn.Module):
    """Combined loss with MSE and SNR components."""
    
    def __init__(self, snr_weight: float = 1.0, mse_weight: float = 0.1):
        super().__init__()
        self.snr_loss = SNRLoss()
        self.mse_weight = mse_weight
        self.snr_weight = snr_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Compute SNR loss (handles NaNs and mask internally)
        snr_loss_val = self.snr_loss(pred, target, mask)
        
        # Handle NaNs for MSE loss
        valid_mask = ~torch.isnan(target)
        if mask is not None:
            valid_mask = valid_mask & mask.unsqueeze(1).bool()
            
        # Masked MSE
        target_filled = torch.where(valid_mask, target, torch.zeros_like(target))
        pred_filled = torch.where(valid_mask, pred, torch.zeros_like(pred))
        
        mse = F.mse_loss(pred_filled, target_filled, reduction='sum')
        num_valid = valid_mask.sum()
        mse = mse / (num_valid + 1e-8)
        
        return self.snr_weight * snr_loss_val + self.mse_weight * mse
