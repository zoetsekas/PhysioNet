"""
Segmentation loss functions for ECG digitization.

Includes Dice Loss and combined losses for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks.
    
    Handles extreme class imbalance (signal pixels are <1% of image).
    """
    
    def __init__(self, smooth: float = 1e-5):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss.
        
        Args:
            pred: Predictions [B, C, H, W] (logits or probabilities)
            target: Ground truth [B, C, H, W] (binary)
            
        Returns:
            Scalar loss value
        """
        # Apply sigmoid if logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - Dice)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focuses on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss.
        
        Args:
            pred: Predictions [B, C, H, W] (logits)
            target: Ground truth [B, C, H, W] (binary)
            
        Returns:
            Scalar loss value
        """
        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute probability
        prob = torch.sigmoid(pred)
        
        # Focal weight
        pt = torch.where(target == 1, prob, 1 - prob)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()


class SegmentationLoss(nn.Module):
    """Combined loss for segmentation.
    
    Combines Dice Loss and Cross-Entropy for better training stability.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        use_focal: bool = False,
    ):
        """Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            ce_weight: Weight for CE/Focal loss component
            use_focal: Use Focal Loss instead of BCE
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        
        if use_focal:
            self.ce_loss = FocalLoss()
        else:
            self.ce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            pred: Predictions [B, C, H, W] (logits)
            target: Ground truth [B, C, H, W] (binary)
            
        Returns:
            Scalar loss value
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        return total_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss.
    
    Allows tuning the balance between false positives and false negatives.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-5):
        """Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Tversky Loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
            
        Returns:
            Scalar loss value
        """
        # Apply sigmoid
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky
