"""
Image transforms for ECG image preprocessing.
"""

from typing import Dict, Tuple, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (1024, 1280),
    p_augment: float = 0.5,
) -> A.Compose:
    """Get training transforms with augmentation.
    
    Args:
        image_size: Target (height, width) for resizing
        p_augment: Probability of applying augmentations
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        # Resize to standard size
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Geometric augmentations - simulate scanning/photography variations
        A.OneOf([
            A.SafeRotate(limit=3, border_mode=0, p=1.0),
            A.Perspective(scale=(0.02, 0.05), p=1.0),
            A.Affine(scale=(0.98, 1.02), translate_percent=0.02, p=1.0),
        ], p=p_augment),
        
        # Color/lighting augmentations - simulate different scanners/cameras
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=1.0),
            A.ToGray(p=1.0),  # Some ECGs are grayscale
        ], p=p_augment),
        
        # Blur - simulate focus issues in photos
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=p_augment * 0.3),
        
        # Noise - simulate sensor noise
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
        ], p=p_augment * 0.3),
        
        # Image quality degradation - simulate JPEG artifacts
        A.ImageCompression(quality_lower=70, quality_upper=100, p=p_augment * 0.2),
        
        # Normalize to ImageNet stats (for pretrained backbones)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        
        ToTensorV2(),
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (1024, 1280),
) -> A.Compose:
    """Get validation/inference transforms without augmentation.
    
    Args:
        image_size: Target (height, width) for resizing
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(
    image_size: Tuple[int, int] = (1024, 1280),
) -> list:
    """Get test-time augmentation transforms.
    
    Args:
        image_size: Target (height, width) for resizing
        
    Returns:
        List of transforms for TTA
    """
    base_transform = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
    
    transforms_list = [
        # Original
        A.Compose(base_transform),
        
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            *base_transform,
        ]),
        
        # Slight rotation
        A.Compose([
            A.SafeRotate(limit=2, border_mode=0, p=1.0),
            *base_transform,
        ]),
        
        # Brightness adjustment
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
            *base_transform,
        ]),
    ]
    
    return transforms_list
