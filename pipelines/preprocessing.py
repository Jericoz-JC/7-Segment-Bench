"""Shared image preprocessing utilities for CV pipelines."""

import cv2
import numpy as np
from pipelines.base import ROI


def crop_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    """Crop image to ROI bounds, clamped to image dimensions."""
    h, w = image.shape[:2]
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(w, roi.x + roi.width)
    y2 = min(h, roi.y + roi.height)
    return image[y1:y2, x1:x2]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale, handling already-gray images."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_height(image: np.ndarray, target_height: int = 100) -> np.ndarray:
    """Resize image to target height preserving aspect ratio."""
    h, w = image.shape[:2]
    if h == 0:
        return image
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)


def apply_morphology(binary: np.ndarray, op: str = 'close', ksize: int = 3) -> np.ndarray:
    """Apply morphological operation to clean up binary image.

    Args:
        binary: Binary image
        op: 'close', 'open', 'dilate', 'erode'
        ksize: Kernel size
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    ops = {
        'close': cv2.MORPH_CLOSE,
        'open': cv2.MORPH_OPEN,
    }
    if op in ops:
        return cv2.morphologyEx(binary, ops[op], kernel)
    elif op == 'dilate':
        return cv2.dilate(binary, kernel, iterations=1)
    elif op == 'erode':
        return cv2.erode(binary, kernel, iterations=1)
    return binary


def invert_if_dark(binary: np.ndarray) -> np.ndarray:
    """Ensure digits are white on black background.

    If more than half the pixels are white, invert.
    """
    white_ratio = np.count_nonzero(binary) / binary.size
    if white_ratio > 0.5:
        return cv2.bitwise_not(binary)
    return binary
