"""Metrics for ECG signal evaluation."""

import numpy as np
from typing import Dict, Optional


def compute_snr(
    pred: np.ndarray,
    target: np.ndarray,
    max_shift: float = 0.2,
    fs: int = 500,
) -> float:
    """Compute Signal-to-Noise Ratio with alignment.
    
    Args:
        pred: Predicted signal [T]
        target: Ground truth signal [T]
        max_shift: Maximum time shift in seconds
        fs: Sampling frequency
        
    Returns:
        SNR in decibels
    """
    max_samples = int(max_shift * fs)
    
    best_snr = -np.inf
    
    for shift in range(-max_samples, max_samples + 1):
        if shift > 0:
            p, t = pred[shift:], target[:-shift]
        elif shift < 0:
            p, t = pred[:shift], target[-shift:]
        else:
            p, t = pred, target
        
        min_len = min(len(p), len(t))
        p, t = p[:min_len], t[:min_len]
        
        # Optimal vertical offset
        offset = np.mean(t - p)
        p_aligned = p + offset
        
        signal_power = np.sum(t ** 2)
        noise_power = np.sum((t - p_aligned) ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            if snr > best_snr:
                best_snr = snr
    
    return best_snr


def compute_lead_snrs(
    pred: np.ndarray,
    target: np.ndarray,
    leads: list = None,
    fs: int = 500,
) -> Dict[str, float]:
    """Compute SNR for each lead."""
    if leads is None:
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    snrs = {}
    for i, lead in enumerate(leads):
        snrs[lead] = compute_snr(pred[i], target[i], fs=fs)
    
    snrs["mean"] = np.mean(list(snrs.values()))
    return snrs
