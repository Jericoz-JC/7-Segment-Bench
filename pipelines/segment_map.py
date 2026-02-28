"""7-segment geometry mapping and digit classification.

Defines segment layout as proportional regions within a digit bounding box,
plus a truth table mapping segment on/off combinations to digits 0-9.

Segment layout:
     _a_
    |   |
    f   b
    |_g_|
    |   |
    e   c
    |_d_|

Used by pipelines 1-3 for pixel-counting-based digit recognition.
"""

import cv2
import numpy as np

# Render regions: used for drawing synthetic digits
# (y_start_frac, y_end_frac, x_start_frac, x_end_frac) relative to bounding box
SEGMENT_REGIONS = {
    'a': (0.00, 0.13, 0.15, 0.85),  # top horizontal
    'b': (0.05, 0.45, 0.72, 0.98),  # top-right vertical
    'c': (0.55, 0.95, 0.72, 0.98),  # bottom-right vertical
    'd': (0.87, 1.00, 0.15, 0.85),  # bottom horizontal
    'e': (0.55, 0.95, 0.02, 0.28),  # bottom-left vertical
    'f': (0.05, 0.45, 0.02, 0.28),  # top-left vertical
    'g': (0.44, 0.56, 0.15, 0.85),  # middle horizontal
}

# Sample regions: smaller non-overlapping areas for fill-ratio measurement.
# Horizontals sample from center x-band (avoiding vertical segment columns).
# Verticals sample from center y-band (avoiding horizontal segment rows).
SAMPLE_REGIONS = {
    'a': (0.02, 0.10, 0.30, 0.70),  # top horiz, center strip
    'b': (0.15, 0.38, 0.76, 0.96),  # top-right vert, core
    'c': (0.62, 0.85, 0.76, 0.96),  # bottom-right vert, core
    'd': (0.90, 0.98, 0.30, 0.70),  # bottom horiz, center strip
    'e': (0.62, 0.85, 0.04, 0.24),  # bottom-left vert, core
    'f': (0.15, 0.38, 0.04, 0.24),  # top-left vert, core
    'g': (0.46, 0.54, 0.35, 0.65),  # middle horiz, narrow center
}

# Truth table: which segments are ON for each digit
DIGIT_SEGMENTS = {
    0: {'a', 'b', 'c', 'd', 'e', 'f'},
    1: {'b', 'c'},
    2: {'a', 'b', 'd', 'e', 'g'},
    3: {'a', 'b', 'c', 'd', 'g'},
    4: {'b', 'c', 'f', 'g'},
    5: {'a', 'c', 'd', 'f', 'g'},
    6: {'a', 'c', 'd', 'e', 'f', 'g'},
    7: {'a', 'b', 'c'},
    8: {'a', 'b', 'c', 'd', 'e', 'f', 'g'},
    9: {'a', 'b', 'c', 'd', 'f', 'g'},
}

# Inverse: frozenset of segments -> digit
SEGMENTS_TO_DIGIT = {frozenset(v): k for k, v in DIGIT_SEGMENTS.items()}


def extract_segment_values(binary_digit: np.ndarray) -> dict[str, float]:
    """Given a binary (thresholded) image of a single digit, compute
    the fill ratio for each of the 7 segments.

    Args:
        binary_digit: Binary image (white=on, black=off) of a single digit

    Returns:
        Dict mapping segment name to fill ratio (0.0 to 1.0)
    """
    h, w = binary_digit.shape[:2]
    values = {}
    for seg_name, (y1f, y2f, x1f, x2f) in SAMPLE_REGIONS.items():
        y1 = int(y1f * h)
        y2 = max(int(y2f * h), y1 + 1)
        x1 = int(x1f * w)
        x2 = max(int(x2f * w), x1 + 1)
        region = binary_digit[y1:y2, x1:x2]
        total = region.size
        if total == 0:
            values[seg_name] = 0.0
        else:
            values[seg_name] = float(np.count_nonzero(region)) / total
    return values


def classify_digit(segment_values: dict[str, float], threshold: float = 0.35) -> tuple[str, float]:
    """Classify a digit from segment fill ratios.

    Args:
        segment_values: Dict of segment name -> fill ratio
        threshold: Fill ratio threshold above which a segment is considered ON

    Returns:
        (digit_string, confidence) — digit as str, confidence 0-1
    """
    on_segments = frozenset(
        seg for seg, val in segment_values.items() if val >= threshold
    )

    if on_segments in SEGMENTS_TO_DIGIT:
        return str(SEGMENTS_TO_DIGIT[on_segments]), 1.0

    # No exact match — find closest digit by segment overlap
    best_digit = '?'
    best_score = -1
    for digit, expected in DIGIT_SEGMENTS.items():
        expected_set = frozenset(expected)
        intersection = len(on_segments & expected_set)
        union = len(on_segments | expected_set)
        score = intersection / union if union > 0 else 0
        if score > best_score:
            best_score = score
            best_digit = str(digit)

    return best_digit, best_score


def _merge_rects(rects: list[tuple], gap_threshold: float) -> list[tuple]:
    """Merge bounding rectangles whose x-ranges overlap or are within gap_threshold."""
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: r[0])
    merged = [list(rects[0])]
    for x, y, cw, ch in rects[1:]:
        prev = merged[-1]
        prev_x2 = prev[0] + prev[2]
        # If this rect overlaps or is close to the previous one
        if x <= prev_x2 + gap_threshold:
            # Merge
            new_x = min(prev[0], x)
            new_y = min(prev[1], y)
            new_x2 = max(prev_x2, x + cw)
            new_y2 = max(prev[1] + prev[3], y + ch)
            merged[-1] = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
        else:
            merged.append([x, y, cw, ch])
    return [tuple(r) for r in merged]


def isolate_digits(binary_image: np.ndarray, num_digits: int = 0) -> list[np.ndarray]:
    """Isolate individual digit regions from a binary image of a display.

    Uses contour analysis with x-overlap merging and fallback to equal-width subdivision.

    Args:
        binary_image: Binary threshold image of the display ROI
        num_digits: Expected number of digits (0 = auto-detect)

    Returns:
        List of cropped binary images, one per digit, left-to-right
    """
    h, w = binary_image.shape[:2]

    # When num_digits is known, use equal-width subdivision for reliable
    # segment geometry (avoids tight-crop distortion on narrow digits like "1").
    # Trim only empty top/bottom bands first so sample regions align with segments.
    if num_digits > 0:
        work_image = binary_image
        row_counts = np.count_nonzero(binary_image, axis=1)
        min_active = max(1, int(0.01 * w))
        active_rows = np.where(row_counts >= min_active)[0]
        if active_rows.size > 0:
            pad = 2
            y1 = max(0, int(active_rows[0]) - pad)
            y2 = min(h, int(active_rows[-1]) + pad + 1)
            # Guard against accidental over-cropping from sparse noise.
            if (y2 - y1) >= max(5, int(0.4 * h)):
                work_image = binary_image[y1:y2, :]

        digit_w = work_image.shape[1] / num_digits
        digits = []
        for i in range(num_digits):
            x1 = int(i * digit_w)
            x2 = int((i + 1) * digit_w)
            digits.append(work_image[:, x1:x2])
        return digits

    # Auto-detect: try contour-based isolation
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by minimum size
    min_h = h * 0.15
    min_w = w * 0.02
    rects = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch >= min_h and cw >= min_w:
            rects.append((x, y, cw, ch))

    # Merge overlapping/adjacent rects into digit groups
    gap = w * 0.03  # Allow small gap for disconnected segments
    digit_rects = _merge_rects(rects, gap)

    # Sort left to right
    digit_rects.sort(key=lambda r: r[0])

    if len(digit_rects) >= 1:
        digits = []
        # Enforce minimum aspect ratio for narrow digits
        min_aspect = 0.5
        for x, y, cw, ch in digit_rects:
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            crop = binary_image[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]
            min_w_for_digit = int(crop_h * min_aspect)
            if crop_w < min_w_for_digit:
                needed = min_w_for_digit - crop_w
                left_pad = needed // 2
                right_pad = needed - left_pad
                crop = cv2.copyMakeBorder(crop, 0, 0, left_pad, right_pad,
                                          cv2.BORDER_CONSTANT, value=0)
            digits.append(crop)
        return digits

    # Fallback: equal-width subdivision
    n = num_digits if num_digits > 0 else max(1, w // (h * 0.6))
    n = int(n)
    digit_w = w / n
    digits = []
    for i in range(n):
        x1 = int(i * digit_w)
        x2 = int((i + 1) * digit_w)
        digits.append(binary_image[:, x1:x2])
    return digits


def read_display(binary_image: np.ndarray, num_digits: int = 0,
                 seg_threshold: float = 0.35) -> tuple[str, float]:
    """Full pipeline: isolate digits and classify each one.

    Returns:
        (digit_string, mean_confidence)
    """
    digits = isolate_digits(binary_image, num_digits)
    if not digits:
        return '', 0.0

    result = ''
    total_conf = 0.0
    for digit_img in digits:
        seg_vals = extract_segment_values(digit_img)
        ch, conf = classify_digit(seg_vals, seg_threshold)
        result += ch
        total_conf += conf

    mean_conf = total_conf / len(digits) if digits else 0.0
    return result, mean_conf
