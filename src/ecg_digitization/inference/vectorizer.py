"""
Column-wise vectorization for converting segmentation masks to 1D signals.

Based on SignalSavants winning approach from PhysioNet 2024 Challenge.
Simple and robust centroid-based extraction.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import interpolate
import logging


def vectorize_mask(
    mask: np.ndarray,
    method: str = "centroid",
) -> np.ndarray:
    """Convert binary segmentation mask to 1D signal via column-wise extraction.
    
    Args:
        mask: Binary mask [H, W] where 1 = signal pixel, 0 = background
        method: Extraction method ("centroid", "median", "weighted")
        
    Returns:
        1D signal as array of y-coordinates [W]
    """
    height, width = mask.shape
    signal_1d = []
    
    for col in range(width):
        # Find all signal pixels in this column
        signal_pixels = np.where(mask[:, col] > 0.5)[0]  # Handle probabilistic masks
        
        if len(signal_pixels) == 0:
            # No signal in this column (gap)
            signal_1d.append(np.nan)
        elif method == "centroid":
            # Average y-coordinate (centroid)
            centroid_y = np.mean(signal_pixels)
            signal_1d.append(centroid_y)
        elif method == "median":
            # Median y-coordinate (robust to outliers)
            median_y = np.median(signal_pixels)
            signal_1d.append(median_y)
        elif method == "weighted":
            # Weighted by mask probability (if mask is probabilistic)
            weights = mask[signal_pixels, col]
            weighted_y = np.average(signal_pixels, weights=weights)
            signal_1d.append(weighted_y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    signal_1d = np.array(signal_1d)
    
    # Count gaps
    num_gaps = np.isnan(signal_1d).sum()
    gap_percent = 100 * num_gaps / len(signal_1d)
    
    if gap_percent > 10:
        logging.getLogger(__name__).warning(f"Signal has {gap_percent:.1f}% gaps ({num_gaps}/{len(signal_1d)})")
    
    # Fill gaps via interpolation
    signal_1d = interpolate_gaps(signal_1d)
    
    return signal_1d


def interpolate_gaps(
    signal: np.ndarray,
    method: str = "linear",
    max_gap_size: int = 50,
) -> np.ndarray:
    """Fill NaN gaps in signal via interpolation.
    
    Args:
        signal: 1D array with potential NaN values
        method: Interpolation method ("linear", "cubic", "nearest")
        max_gap_size: Maximum gap size to interpolate (larger gaps use nearest)
        
    Returns:
        Signal with gaps filled
    """
    signal = signal.copy()
    
    # Find valid (non-NaN) indices
    valid_mask = ~np.isnan(signal)
    valid_indices = np.where(valid_mask)[0]
    valid_values = signal[valid_mask]
    
    if len(valid_indices) == 0:
        logging.getLogger(__name__).error("Signal is entirely NaN!")
        return np.zeros_like(signal)
    
    if len(valid_indices) == len(signal):
        return signal  # No gaps
    
    # Find gap regions
    gaps = find_gap_regions(valid_mask)
    
    # Interpolate each gap
    for gap_start, gap_end in gaps:
        gap_size = gap_end - gap_start
        
        if gap_size > max_gap_size:
            # Large gap: use nearest neighbor
            logging.getLogger(__name__).warning(f"Large gap ({gap_size} pixels) at {gap_start}, using nearest")
            if gap_start > 0:
                signal[gap_start:gap_end] = signal[gap_start - 1]
            elif gap_end < len(signal):
                signal[gap_start:gap_end] = signal[gap_end]
        else:
            # Small gap: interpolate
            if method == "linear":
                interp_func = interpolate.interp1d(
                    valid_indices, valid_values,
                    kind='linear',
                    fill_value='extrapolate'
                )
            elif method == "cubic":
                interp_func = interpolate.interp1d(
                    valid_indices, valid_values,
                    kind='cubic',
                    fill_value='extrapolate'
                )
            else:
                interp_func = interpolate.interp1d(
                    valid_indices, valid_values,
                    kind='nearest',
                    fill_value='extrapolate'
                )
            
            gap_indices = np.arange(gap_start, gap_end)
            signal[gap_indices] = interp_func(gap_indices)
    
    return signal


def find_gap_regions(valid_mask: np.ndarray) -> list:
    """Find contiguous regions of False in boolean mask.
    
    Args:
        valid_mask: Boolean array where True = valid pixel
        
    Returns:
        List of (start, end) tuples for each gap region
    """
    gaps = []
    in_gap = False
    gap_start = None
    
    for i, is_valid in enumerate(valid_mask):
        if not is_valid and not in_gap:
            # Gap start
            gap_start = i
            in_gap = True
        elif is_valid and in_gap:
            # Gap end
            gaps.append((gap_start, i))
            in_gap = False
    
    # Handle gap extending to end
    if in_gap:
        gaps.append((gap_start, len(valid_mask)))
    
    return gaps


def extract_multi_lead_signals(
    mask: np.ndarray,
    layout: str = "3x4",
    num_leads: int = 12,
) -> np.ndarray:
    """Extract multiple lead signals from a composite mask.
    
    Args:
        mask: Full ECG segmentation mask [H, W]
        layout: ECG layout format ("3x4", "6x2", "12x1")
        num_leads: Number of leads to extract
        
    Returns:
        Array of signals [num_leads, T]
    """
    height, width = mask.shape
    
    if layout == "3x4":
        # 3 rows, 4 columns (standard 12-lead)
        row_height = height // 4  # 4 rows (3 main + 1 rhythm strip)
        signals = []
        
        # Extract 3x4 grid
        for row in range(3):
            for col in range(4):
                y_start = row * row_height
                y_end = (row + 1) * row_height
                
                lead_mask = mask[y_start:y_end, :]
                signal = vectorize_mask(lead_mask)
                signals.append(signal)
        
        return np.array(signals[:num_leads])
    
    elif layout == "6x2":
        # 6 rows, 2 columns
        row_height = height // 6
        signals = []
        
        for row in range(6):
            y_start = row * row_height
            y_end = (row + 1) * row_height
            
            # Left half
            left_mask = mask[y_start:y_end, :width//2]
            signals.append(vectorize_mask(left_mask))
            
            # Right half
            right_mask = mask[y_start:y_end, width//2:]
            signals.append(vectorize_mask(right_mask))
        
        return np.array(signals[:num_leads])
    
    else:
        # Single column layout
        lead_height = height // num_leads
        signals = []
        
        for lead_idx in range(num_leads):
            y_start = lead_idx * lead_height
            y_end = (lead_idx + 1) * lead_height
            
            lead_mask = mask[y_start:y_end, :]
            signal = vectorize_mask(lead_mask)
            signals.append(signal)
        
        return np.array(signals)
