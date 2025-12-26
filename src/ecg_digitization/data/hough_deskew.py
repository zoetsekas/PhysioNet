"""
Hough Transform-based image deskewing for ECG digitization.

Based on SignalSavants winning approach from PhysioNet 2024 Challenge.
Grid lines are detected via Hough Transform and the image is rotated
to align the time axis horizontally.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from loguru import logger


def deskew_ecg_image(
    image: np.ndarray,
    edge_threshold1: int = 50,
    edge_threshold2: int = 150,
    hough_threshold: int = 200,
    angle_tolerance: float = 45.0,
) -> Tuple[np.ndarray, float]:
    """Detect and correct rotation using Hough Transform on grid lines.
    
    Args:
        image: Input image [H, W, C] or [H, W]
        edge_threshold1: Lower threshold for Canny edge detection
        edge_threshold2: Upper threshold for Canny edge detection
        hough_threshold: Threshold for Hough line detection
        angle_tolerance: Maximum rotation angle to consider (degrees)
        
    Returns:
        Tuple of (deskewed_image, rotation_angle_degrees)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection with Canny
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=hough_threshold)
    
    if lines is None or len(lines) == 0:
        logger.warning("No lines detected via Hough Transform, returning original image")
        return image, 0.0
    
    # Extract angles from detected lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        
        # Normalize to [-90, 90] range
        if angle_deg > 90:
            angle_deg -= 180
        
        # Filter out extreme angles
        if abs(angle_deg) < angle_tolerance:
            angles.append(angle_deg)
    
    if len(angles) == 0:
        logger.warning("No valid angles found, returning original image")
        return image, 0.0
    
    # Find dominant angle (median is robust to outliers)
    dominant_angle = np.median(angles)
    
    logger.info(f"Detected rotation: {dominant_angle:.2f} degrees from {len(angles)} grid lines")
    
    # Rotate image to correct alignment
    deskewed = rotate_image(image, -dominant_angle)
    
    return deskewed, dominant_angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated image with same dimensions
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    return rotated


def detect_grid_orientation_robust(
    image: np.ndarray,
    num_bins: int = 180,
) -> float:
    """Alternative method: Use histogram of edge gradients.
    
    This is more robust when grid is faint or partially occluded.
    
    Args:
        image: Input image
        num_bins: Number of angle bins for histogram
        
    Returns:
        Dominant angle in degrees
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient angles
    angles = np.arctan2(gy, gx)
    angles_deg = np.degrees(angles).flatten()
    
    # Build histogram
    hist, bin_edges = np.histogram(
        angles_deg,
        bins=num_bins,
        range=(-90, 90),
        weights=np.abs(gx.flatten()) + np.abs(gy.flatten())  # Weight by gradient magnitude
    )
    
    # Find dominant angle
    peak_idx = np.argmax(hist)
    dominant_angle = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    
    return dominant_angle


class HoughDeskewer:
    """Stateful deskewer with caching and fallback strategies."""
    
    def __init__(
        self,
        primary_method: str = "hough",
        fallback_method: str = "gradient",
    ):
        """Initialize deskewer.
        
        Args:
            primary_method: Primary detection method ("hough" or "gradient")
            fallback_method: Fallback if primary fails
        """
        self.primary_method = primary_method
        self.fallback_method = fallback_method
        self.last_angle = 0.0
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew image with fallback.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (deskewed_image, angle)
        """
        try:
            if self.primary_method == "hough":
                deskewed, angle = deskew_ecg_image(image)
                if abs(angle) < 1e-3:  # Failed
                    raise ValueError("Hough failed")
            else:
                angle = detect_grid_orientation_robust(image)
                deskewed = rotate_image(image, -angle)
        except Exception as e:
            logger.warning(f"Primary method failed: {e}, using fallback")
            
            if self.fallback_method == "gradient":
                angle = detect_grid_orientation_robust(image)
                deskewed = rotate_image(image, -angle)
            else:
                # Last resort: use previous angle
                logger.warning("Fallback failed, using last known angle")
                angle = self.last_angle
                deskewed = rotate_image(image, -angle)
        
        self.last_angle = angle
        return deskewed, angle
