"""
Input transformation defenses.

These preprocessing-based defenses modify inputs before classification,
aiming to remove adversarial perturbations while preserving legitimate
signal. They are model-agnostic and can be applied at inference time
without retraining.

Implements:
- Feature squeezing (Xu et al., 2018): reduce feature precision
- Spatial smoothing: local averaging over feature windows
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def feature_squeezing(X: np.ndarray, bit_depth: int = 4) -> np.ndarray:
    """
    Reduce feature precision by quantizing to a lower bit depth.

    For standardized features (roughly in [-3, 3]), we map to [0, 1],
    quantize, then map back. This destroys small adversarial perturbations
    while preserving the coarse signal.

    Args:
        X:         Input features (N, D), float32.
        bit_depth: Number of bits for quantization (fewer = more squeezing).

    Returns:
        Squeezed features.
    """
    levels = 2 ** bit_depth
    # Soft normalization to [0, 1] range
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    x_range = np.maximum(x_max - x_min, 1e-8)

    X_norm = (X - x_min) / x_range
    X_quantized = np.round(X_norm * levels) / levels
    X_squeezed = X_quantized * x_range + x_min

    logger.debug("Feature squeezing: bit_depth=%d, levels=%d", bit_depth, levels)
    return X_squeezed.astype(np.float32)


def spatial_smoothing(X: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply 1D moving average over the feature dimension.

    For tabular data, this acts as a local feature smoother that can
    disrupt the precise gradient-aligned perturbations used by FGSM/PGD.

    Args:
        X:           Input features (N, D).
        window_size: Size of the averaging window.

    Returns:
        Smoothed features.
    """
    if window_size < 2:
        return X

    N, D = X.shape
    pad = window_size // 2
    X_padded = np.pad(X, ((0, 0), (pad, pad)), mode="edge")
    X_smooth = np.zeros_like(X)

    for i in range(D):
        X_smooth[:, i] = X_padded[:, i:i + window_size].mean(axis=1)

    logger.debug("Spatial smoothing: window_size=%d", window_size)
    return X_smooth.astype(np.float32)


def apply_input_defense(X: np.ndarray, method: str = "squeezing",
                        **kwargs) -> np.ndarray:
    """
    Convenience dispatcher for input transformation defenses.

    Args:
        X:      Input features.
        method: "squeezing" or "smoothing".
        **kwargs: Passed to the selected defense.

    Returns:
        Transformed features.
    """
    if method == "squeezing":
        return feature_squeezing(X, **kwargs)
    elif method == "smoothing":
        return spatial_smoothing(X, **kwargs)
    else:
        raise ValueError(f"Unknown input defense method: {method}")
