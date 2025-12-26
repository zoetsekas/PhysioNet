"""
ECG image preprocessing utilities.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def preprocess_ecg_image(
    image: np.ndarray,
    remove_grid: bool = True,
    enhance_signal: bool = True,
    binarize: bool = False,
) -> np.ndarray:
    """Preprocess ECG image for better signal extraction.
    
    Args:
        image: Input image [H, W, C] in RGB
        remove_grid: Whether to remove the grid background
        enhance_signal: Whether to enhance signal lines
        binarize: Whether to convert to binary image
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Remove grid if requested
    if remove_grid:
        gray = _remove_grid_background(gray)
    
    # Enhance signal lines
    if enhance_signal:
        gray = _enhance_signal_lines(gray)
    
    # Binarize if requested
    if binarize:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to 3-channel if input was RGB
    if len(image.shape) == 3:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return gray


def _remove_grid_background(image: np.ndarray) -> np.ndarray:
    """Remove the ECG grid background.
    
    The grid is typically pink/red on white background. We want to remove
    the grid lines while preserving the black signal lines.
    
    Args:
        image: Grayscale image
        
    Returns:
        Image with grid removed
    """
    # Apply morphological operations to remove thin grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Opening removes small details (grid lines)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Closing fills small gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def _enhance_signal_lines(image: np.ndarray) -> np.ndarray:
    """Enhance the ECG signal lines.
    
    Args:
        image: Grayscale image
        
    Returns:
        Image with enhanced signal lines
    """
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Sharpen the image
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def extract_grid_info(image: np.ndarray) -> Dict[str, float]:
    """Extract grid information from ECG image.
    
    Standard ECG paper:
    - Small squares: 1mm x 1mm
    - Large squares: 5mm x 5mm (5 small squares)
    - Paper speed: 25mm/s (each small square = 0.04s)
    - Amplitude: 10mm/mV (each small square = 0.1mV)
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with grid information:
            - pixels_per_mm: Pixels per millimeter
            - pixels_per_small_square: Pixels per small grid square
            - pixels_per_large_square: Pixels per large grid square
            - time_per_pixel: Time (seconds) per pixel
            - amplitude_per_pixel: Amplitude (mV) per pixel
    """
    # Convert to HSV for better color detection
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        # Create dummy HSV for grayscale
        hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2HSV)
    
    # Detect red/pink grid lines
    # Red color range in HSV
    lower_red1 = np.array([0, 30, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 30, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    grid_mask = mask1 + mask2
    
    # Find grid line spacing using FFT or peak detection
    # Horizontal projection (sum along rows)
    h_projection = np.sum(grid_mask, axis=1)
    
    # Vertical projection (sum along columns)
    v_projection = np.sum(grid_mask, axis=0)
    
    # Find peaks in projections
    h_spacing = _find_grid_spacing(h_projection)
    v_spacing = _find_grid_spacing(v_projection)
    
    # Average spacing (should be approximately equal for square grid)
    avg_small_spacing = (h_spacing + v_spacing) / 2 if h_spacing and v_spacing else 20.0  # Default
    
    # Standard ECG paper: small square = 1mm
    pixels_per_mm = avg_small_spacing
    
    return {
        "pixels_per_mm": pixels_per_mm,
        "pixels_per_small_square": avg_small_spacing,
        "pixels_per_large_square": avg_small_spacing * 5,
        "time_per_pixel": 0.04 / avg_small_spacing,  # 1mm = 0.04s at 25mm/s
        "amplitude_per_pixel": 0.1 / avg_small_spacing,  # 1mm = 0.1mV at 10mm/mV
    }


def _find_grid_spacing(projection: np.ndarray) -> Optional[float]:
    """Find grid line spacing from a 1D projection.
    
    Args:
        projection: 1D array of projection values
        
    Returns:
        Average spacing between grid lines in pixels
    """
    from scipy import signal
    
    # Normalize
    projection = projection.astype(float)
    projection = (projection - projection.min()) / (projection.max() - projection.min() + 1e-8)
    
    # Find peaks (grid lines)
    peaks, properties = signal.find_peaks(projection, height=0.3, distance=10)
    
    if len(peaks) < 2:
        return None
    
    # Calculate average spacing
    spacings = np.diff(peaks)
    
    # Filter outliers (keep spacings close to median)
    median_spacing = np.median(spacings)
    valid_spacings = spacings[np.abs(spacings - median_spacing) < median_spacing * 0.3]
    
    if len(valid_spacings) > 0:
        return float(np.mean(valid_spacings))
    
    return float(median_spacing)


def detect_leads(image: np.ndarray) -> List[Dict[str, int]]:
    """Detect ECG lead regions in the image.
    
    Standard 12-lead ECG layout:
    - 4 rows of 3 leads each
    - Row 1: I, aVR, V1, V4
    - Row 2: II, aVL, V2, V5
    - Row 3: III, aVF, V3, V6
    - Row 4: Continuous lead II (rhythm strip)
    
    Args:
        image: Input image
        
    Returns:
        List of dictionaries with lead bounding boxes:
            - lead: Lead name
            - x, y, w, h: Bounding box
    """
    height, width = image.shape[:2]
    
    # Standard 4-row layout
    row_height = height // 4
    lead_width = width // 4 if width > height else width // 3
    
    leads = []
    
    # Define lead positions (standard layout)
    lead_layout = [
        # Row 0
        [("I", 0), ("aVR", 1), ("V1", 2), ("V4", 3)],
        # Row 1
        [("II", 0), ("aVL", 1), ("V2", 2), ("V5", 3)],
        # Row 2
        [("III", 0), ("aVF", 1), ("V3", 2), ("V6", 3)],
        # Row 3 - rhythm strip (usually lead II)
        [("II_rhythm", 0)],
    ]
    
    for row_idx, row_leads in enumerate(lead_layout):
        y = row_idx * row_height
        h = row_height
        
        for lead_name, col_idx in row_leads:
            if row_idx == 3:
                # Rhythm strip spans full width
                x, w = 0, width
            else:
                x = col_idx * lead_width
                w = lead_width
            
            leads.append({
                "lead": lead_name,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })
    
    return leads


def extract_signal_from_lead(
    image: np.ndarray,
    lead_bbox: Dict[str, int],
    grid_info: Dict[str, float],
) -> np.ndarray:
    """Extract time-series signal from a single lead region.
    
    Args:
        image: Full ECG image
        lead_bbox: Lead bounding box dictionary
        grid_info: Grid calibration info from extract_grid_info
        
    Returns:
        1D numpy array of signal values in mV
    """
    # Crop lead region
    x, y, w, h = lead_bbox["x"], lead_bbox["y"], lead_bbox["w"], lead_bbox["h"]
    lead_img = image[y:y+h, x:x+w]
    
    # Convert to grayscale if needed
    if len(lead_img.shape) == 3:
        gray = cv2.cvtColor(lead_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = lead_img.copy()
    
    # Preprocess to enhance signal
    processed = preprocess_ecg_image(gray, remove_grid=True, enhance_signal=True)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # For each column, find the centroid of the signal line
    signal = []
    baseline = h // 2  # Assume baseline is at center
    
    for col in range(w):
        column = binary[:, col]
        signal_pixels = np.where(column > 0)[0]
        
        if len(signal_pixels) > 0:
            # Use centroid of signal pixels
            centroid = np.mean(signal_pixels)
            # Convert to voltage (positive = up, negative = down)
            voltage = (baseline - centroid) * grid_info["amplitude_per_pixel"]
        else:
            voltage = 0.0
            
        signal.append(voltage)
    
    return np.array(signal)
