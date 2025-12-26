"""
Enhanced multi-method calibration for ECG digitization.

Implements calibration pulse detection, grid-based calibration,
and novel blind calibration using QRS amplitude estimation.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import cv2
from scipy import signal as scipy_signal
import logging


def calibrate_signal(
    pixel_signal: np.ndarray,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fs: int = 500,
) -> np.ndarray:
    """Convert pixel coordinates to millivolts using multiple calibration methods.
    
    Args:
        pixel_signal: 1D signal in pixel coordinates [T]
        image: Original ECG image
        mask: Optional segmentation mask
        fs: Sampling frequency (Hz)
        
    Returns:
        Calibrated signal in millivolts [T]
    """
    # Try calibration methods in order of preference
    pixels_per_mv = None
    method_used = None
    
    # Method 1: Detect calibration pulse (most accurate)
    pixels_per_mv = detect_calibration_pulse(image, mask)
    if pixels_per_mv is not None:
        method_used = "calibration_pulse"
        logging.getLogger(__name__).info(f"Using calibration pulse: {pixels_per_mv:.2f} pixels/mV")
    
    # Method 2: Measure grid spacing
    if pixels_per_mv is None:
        pixels_per_mv = detect_grid_spacing(image)
        if pixels_per_mv is not None:
            method_used = "grid_spacing"
            logging.getLogger(__name__).info(f"Using grid spacing: {pixels_per_mv:.2f} pixels/mV")
    
    # Method 3: Blind calibration from QRS amplitude
    if pixels_per_mv is None:
        pixels_per_mv = estimate_from_qrs_amplitude(pixel_signal, fs)
        method_used = "blind_qrs"
        logging.getLogger(__name__).warning(f"Using blind QRS calibration: {pixels_per_mv:.2f} pixels/mV")
    
    # Detect baseline
    baseline = detect_baseline(pixel_signal)
    
    # Convert to mV
    voltage = (baseline - pixel_signal) / pixels_per_mv  # Invert Y-axis
    
    logging.getLogger(__name__).info(f"Calibration: method={method_used}, baseline={baseline:.1f}px, "
                f"scale={pixels_per_mv:.2f}px/mV, range=[{voltage.min():.2f}, {voltage.max():.2f}]mV")
    
    return voltage


def detect_calibration_pulse(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pulse_width_mm: float = 5.0,
    pulse_height_mv: float = 1.0,
) -> Optional[float]:
    """Detect the standard 1mV calibration pulse (rectangle).
    
    Args:
        image: ECG image
        mask: Optional segmentation mask
        pulse_width_mm: Expected pulse width in mm (typically 5mm = 200ms at 25mm/s)
        pulse_height_mv: Calibration pulse amplitude (typically 1mV)
        
    Returns:
        pixels_per_mv or None if not found
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Find rectangular contours
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for square-ish contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # Calibration pulse criteria:
        # - Roughly square (width ~= height)
        # - Moderate size (not huge, not tiny)
        if 0.8 < aspect_ratio < 1.2 and 100 < area < 10000:
            # This might be the calibration pulse
            pixels_per_mv = h / pulse_height_mv
            logging.getLogger(__name__).info(f"Found potential calibration pulse: {w}x{h}px at ({x},{y})")
            return pixels_per_mv
    
    return None


def detect_grid_spacing(
    image: np.ndarray,
    grid_box_mv: float = 0.5,
) -> Optional[float]:
    """Measure grid spacing to determine scale.
    
    Standard ECG grid: 1 large box (5mm) = 0.5mV
    
    Args:
        image: ECG image
        grid_box_mv: Voltage represented by one large grid box
        
    Returns:
        pixels_per_mv or None if not found
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Detect horizontal grid lines via projection
    horizontal_projection = np.mean(gray, axis=1)
    
    # Find peaks (grid lines are darker than background)
    inverted = 255 - horizontal_projection
    peaks, properties = scipy_signal.find_peaks(
        inverted,
        height=10,
        distance=10,
        prominence=5,
    )
    
    if len(peaks) < 2:
        return None
    
    # Calculate spacing between consecutive peaks
    spacings = np.diff(peaks)
    
    # Filter outliers (keep spacings close to median)
    median_spacing = np.median(spacings)
    valid_spacings = spacings[np.abs(spacings - median_spacing) < median_spacing * 0.3]
    
    if len(valid_spacings) == 0:
        return None
    
    # Average spacing
    avg_spacing = np.mean(valid_spacings)
    
    # Spacing corresponds to 1 large box (0.5 mV)
    pixels_per_mv = avg_spacing / grid_box_mv
    
    logging.getLogger(__name__).info(f"Detected grid spacing: {avg_spacing:.2f} pixels = {grid_box_mv} mV")
    
    return pixels_per_mv


def estimate_from_qrs_amplitude(
    pixel_signal: np.ndarray,
    fs: int = 500,
    expected_r_wave_mv: float = 1.2,
) -> float:
    """Blind calibration using expected QRS complex amplitude.
    
    Uses population-based prior: typical R-wave amplitude is 1.0-1.5 mV.
    
    Args:
        pixel_signal: Signal in pixel coordinates
        fs: Sampling frequency
        expected_r_wave_mv: Expected R-wave amplitude (mV)
        
    Returns:
        Estimated pixels_per_mv
    """
    # Detect QRS complexes (R-peaks)
    r_peaks = detect_r_peaks(pixel_signal, fs)
    
    if len(r_peaks) == 0:
        logging.getLogger(__name__).warning("No R-peaks detected, using default calibration")
        return 20.0  # Fallback default
    
    # Measure R-peak amplitudes (relative to baseline)
    baseline = np.median(pixel_signal)
    r_amplitudes_px = np.abs(pixel_signal[r_peaks] - baseline)
    
    # Use median R-wave amplitude
    median_r_amplitude_px = np.median(r_amplitudes_px)
    
    # Estimate scale
    pixels_per_mv = median_r_amplitude_px / expected_r_wave_mv
    
    logging.getLogger(__name__).info(f"Blind calibration: median R-wave = {median_r_amplitude_px:.1f}px "
                f"→ {pixels_per_mv:.2f} px/mV")
    
    return pixels_per_mv


def detect_r_peaks(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """Detect R-peaks in ECG signal (simple peak detection).
    
    Args:
        signal: 1D ECG signal
        fs: Sampling frequency
        
    Returns:
        Array of R-peak indices
    """
    # Bandpass filter to enhance QRS
    from scipy.signal import butter, filtfilt
    
    # Design bandpass filter (5-15 Hz for QRS)
    nyq = fs / 2
    low = 5 / nyq
    high = 15 / nyq
    b, a = butter(2, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    
    # Square to emphasize peaks
    squared = filtered ** 2
    
    # Find peaks with minimum distance (typical HR: 60-120 bpm)
    min_distance = int(0.6 * fs / 2)  # 0.6s = 100 bpm minimum
    
    peaks, _ = scipy_signal.find_peaks(
        squared,
        distance=min_distance,
        prominence=np.percentile(squared, 75),  # Adaptive threshold
    )
    
    return peaks


def detect_baseline(signal: np.ndarray, percentile: float = 50.0) -> float:
    """Detect baseline (isoelectric line) of ECG signal.
    
    Args:
        signal: 1D signal in pixel coordinates
        percentile: Percentile to use for baseline (50 = median)
        
    Returns:
        Baseline y-coordinate
    """
    # Use median as robust baseline estimator
    baseline = np.percentile(signal, percentile)
    return baseline


def multi_lead_calibration(
    pixel_signals: np.ndarray,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calibrate all 12 leads using shared scale factor.
    
    Args:
        pixel_signals: Multi-lead signals [12, T]
        image: Original image
        mask: Optional mask
        
    Returns:
        Calibrated signals in mV [12, T]
    """
    # Calibrate lead II (usually has best QRS visibility)
    lead_ii_calibrated = calibrate_signal(pixel_signals[1], image, mask)
    
    # Extract scale factor
    baseline = detect_baseline(pixel_signals[1])
    pixels_per_mv = np.abs(pixel_signals[1] - baseline).max() / np.abs(lead_ii_calibrated).max()
    
    # Apply same scale to all leads
    calibrated_signals = []
    for lead_idx, pixel_signal in enumerate(pixel_signals):
        baseline = detect_baseline(pixel_signal)
        voltage = (baseline - pixel_signal) / pixels_per_mv
        calibrated_signals.append(voltage)
    
    return np.array(calibrated_signals)
