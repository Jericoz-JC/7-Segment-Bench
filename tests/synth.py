"""Synthetic 7-segment display image generator for testing."""

import cv2
import numpy as np
from pipelines.segment_map import DIGIT_SEGMENTS, SEGMENT_REGIONS


def render_digit(digit: int, height: int = 120, width: int = 70,
                 fg_color: tuple = (0, 255, 0), bg_color: tuple = (0, 0, 0),
                 thickness: int = 0) -> np.ndarray:
    """Render a single 7-segment digit as a BGR image.

    Args:
        digit: 0-9
        height, width: Image dimensions
        fg_color: BGR color for ON segments
        bg_color: BGR color for background
        thickness: 0 = filled, >0 = outline

    Returns:
        BGR image of the digit
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    segments = DIGIT_SEGMENTS.get(digit, set())

    for seg in segments:
        y1f, y2f, x1f, x2f = SEGMENT_REGIONS[seg]
        y1 = int(y1f * height)
        y2 = int(y2f * height)
        x1 = int(x1f * width)
        x2 = int(x2f * width)
        if thickness == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), fg_color, -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), fg_color, thickness)

    return img


def render_display(digits: str, digit_height: int = 120, digit_width: int = 70,
                   gap: int = 10, fg_color: tuple = (0, 255, 0),
                   bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Render a multi-digit 7-segment display.

    Args:
        digits: String of digits, e.g. "0237"
        digit_height, digit_width: Size per digit
        gap: Pixel gap between digits
        fg_color: BGR for ON segments
        bg_color: BGR for background

    Returns:
        BGR image of the full display
    """
    n = len(digits)
    total_w = n * digit_width + (n - 1) * gap + 20  # 10px padding each side
    total_h = digit_height + 20
    img = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    for i, ch in enumerate(digits):
        if ch.isdigit():
            d_img = render_digit(int(ch), digit_height, digit_width, fg_color, bg_color)
            x_off = 10 + i * (digit_width + gap)
            y_off = 10
            img[y_off:y_off + digit_height, x_off:x_off + digit_width] = d_img

    return img


def add_noise(image: np.ndarray, noise_level: float = 15.0) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_glare(image: np.ndarray, center: tuple = None,
              radius: int = 60, intensity: int = 200) -> np.ndarray:
    """Add a circular glare spot to simulate light reflection."""
    h, w = image.shape[:2]
    if center is None:
        center = (w // 3, h // 2)
    result = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), radius / 3)
    glare = (mask.astype(np.float32) / 255.0 * intensity).astype(np.uint8)
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c].astype(np.int16) + glare, 0, 255).astype(np.uint8)
    return result


def generate_test_images() -> list[tuple[np.ndarray, str]]:
    """Generate a set of test images with known ground truth.

    Returns:
        List of (image, ground_truth) tuples
    """
    test_cases = [
        '0123', '4567', '8900', '1234', '5678',
        '9012', '0000', '1111', '8888', '42',
    ]
    images = []

    for gt in test_cases:
        # Clean image
        img = render_display(gt)
        images.append((img, gt))

        # Noisy version
        noisy = add_noise(img, 20)
        images.append((noisy, gt))

    return images
