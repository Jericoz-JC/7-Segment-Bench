"""Pipeline 3: HSV color filtering with glare rejection -> pixel count."""

import cv2
import numpy as np
from pipelines import register
from pipelines.base import BasePipeline, PipelineResult, ROI
from pipelines.preprocessing import crop_roi, resize_height, invert_if_dark, apply_morphology
from pipelines.segment_map import read_display


@register
class HSVColorPipeline(BasePipeline):
    name = 'HSV Color Filter'
    slug = 'p03_hsv_color'
    description = 'HSV filtering by LED hue -> glare rejection -> pixel count'
    is_online = False
    is_trainable = False

    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        debug = {}

        cropped = crop_roi(image, roi)
        cropped = resize_height(cropped, 100)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Default ranges for common LED colors (red/green)
        # Red LED: hue wraps around 0/180
        color_mode = self.config.get('color', 'red')
        if color_mode == 'green':
            lower = np.array([35, 50, 50])
            upper = np.array([85, 255, 255])
        elif color_mode == 'blue':
            lower = np.array([100, 50, 50])
            upper = np.array([130, 255, 255])
        else:  # red (default)
            # Red wraps: handle both ranges
            lower1 = np.array([0, 50, 50])
            upper1 = np.array([15, 255, 255])
            lower2 = np.array([165, 50, 50])
            upper2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            color_mask = cv2.bitwise_or(mask1, mask2)
            # Skip the standard inRange below
            lower = upper = None

        if lower is not None:
            color_mask = cv2.inRange(hsv, lower, upper)

        debug['color_mask'] = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)

        # Glare rejection: reject low-saturation, high-value pixels
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        glare_mask = (sat < 30) & (val > 200)
        color_mask[glare_mask] = 0
        debug['glare_rejected'] = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)

        # Clean up
        binary = apply_morphology(color_mask, 'close', 3)
        binary = apply_morphology(binary, 'open', 2)
        debug['binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        num_digits = self.config.get('num_digits', 0)
        predicted, confidence = read_display(binary, num_digits)

        return PipelineResult(
            predicted=predicted,
            confidence=confidence,
            debug_images=debug,
        )
