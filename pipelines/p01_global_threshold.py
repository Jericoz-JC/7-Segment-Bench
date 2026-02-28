"""Pipeline 1: Otsu global threshold -> pixel count per segment."""

import cv2
import numpy as np
from pipelines import register
from pipelines.base import BasePipeline, PipelineResult, ROI
from pipelines.preprocessing import crop_roi, to_grayscale, resize_height, invert_if_dark, apply_morphology
from pipelines.segment_map import read_display


@register
class GlobalThresholdPipeline(BasePipeline):
    name = 'Global Threshold (Otsu)'
    slug = 'p01_global_threshold'
    description = 'Otsu threshold -> pixel count per segment'
    is_online = False
    is_trainable = False

    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        debug = {}

        # Crop to ROI
        cropped = crop_roi(image, roi)
        gray = to_grayscale(cropped)
        gray = resize_height(gray, 100)
        debug['grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = invert_if_dark(binary)
        binary = apply_morphology(binary, 'close', 3)
        debug['binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Read digits
        num_digits = self.config.get('num_digits', 0)
        predicted, confidence = read_display(binary, num_digits)

        return PipelineResult(
            predicted=predicted,
            confidence=confidence,
            debug_images=debug,
        )
