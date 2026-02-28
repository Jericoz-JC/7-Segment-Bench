"""Pipeline 2: CLAHE normalization -> Gaussian adaptive threshold -> morphology -> pixel count."""

import cv2
import numpy as np
from pipelines import register
from pipelines.base import BasePipeline, PipelineResult, ROI
from pipelines.preprocessing import crop_roi, to_grayscale, resize_height, invert_if_dark, apply_morphology
from pipelines.segment_map import read_display


@register
class AdaptiveCLAHEPipeline(BasePipeline):
    name = 'Adaptive + CLAHE'
    slug = 'p02_adaptive_clahe'
    description = 'CLAHE normalization -> Gaussian adaptive threshold -> morphology'
    is_online = False
    is_trainable = False

    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        debug = {}

        cropped = crop_roi(image, roi)
        gray = to_grayscale(cropped)
        gray = resize_height(gray, 100)

        # CLAHE contrast normalization
        clip_limit = self.config.get('clip_limit', 3.0)
        tile_size = self.config.get('tile_size', 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(gray)
        debug['clahe'] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Gaussian adaptive threshold
        block_size = self.config.get('block_size', 15)
        if block_size % 2 == 0:
            block_size += 1
        c_val = self.config.get('c_value', 5)
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c_val
        )
        num_digits = self.config.get('num_digits', 0)
        binary_adaptive = invert_if_dark(binary_adaptive)
        binary_adaptive = apply_morphology(binary_adaptive, 'close', 3)
        binary_adaptive = apply_morphology(binary_adaptive, 'open', 2)
        pred_adaptive, conf_adaptive = read_display(binary_adaptive, num_digits)

        # Fallback for block artifacts: global Otsu on CLAHE image.
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_otsu = invert_if_dark(binary_otsu)
        binary_otsu = apply_morphology(binary_otsu, 'close', 3)
        binary_otsu = apply_morphology(binary_otsu, 'open', 2)
        pred_otsu, conf_otsu = read_display(binary_otsu, num_digits)

        use_otsu = False
        if '?' in pred_adaptive and '?' not in pred_otsu:
            use_otsu = True
        elif conf_otsu > (conf_adaptive + 0.05):
            use_otsu = True
        elif conf_adaptive < 0.85 and conf_otsu >= conf_adaptive:
            use_otsu = True

        if use_otsu:
            predicted, confidence = pred_otsu, conf_otsu
            binary = binary_otsu
        else:
            predicted, confidence = pred_adaptive, conf_adaptive
            binary = binary_adaptive

        debug['binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        debug['binary_adaptive'] = cv2.cvtColor(binary_adaptive, cv2.COLOR_GRAY2BGR)
        debug['binary_otsu'] = cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR)

        return PipelineResult(
            predicted=predicted,
            confidence=confidence,
            debug_images=debug,
        )
