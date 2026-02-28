"""Pipeline 4: Normalized cross-correlation template matching."""

import cv2
import numpy as np
from pipelines import register
from pipelines.base import BasePipeline, PipelineResult, ROI
from pipelines.preprocessing import crop_roi, to_grayscale, resize_height, invert_if_dark
from pipelines.segment_map import isolate_digits, DIGIT_SEGMENTS, SEGMENT_REGIONS


def generate_digit_template(digit: int, height: int = 80, width: int = 50) -> np.ndarray:
    """Generate a synthetic 7-segment digit template."""
    img = np.zeros((height, width), dtype=np.uint8)
    segments = DIGIT_SEGMENTS[digit]
    thickness = max(2, width // 8)

    for seg in segments:
        y1f, y2f, x1f, x2f = SEGMENT_REGIONS[seg]
        y1 = int(y1f * height)
        y2 = int(y2f * height)
        x1 = int(x1f * width)
        x2 = int(x2f * width)
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)

    return img


@register
class TemplateMatchingPipeline(BasePipeline):
    name = 'Template Matching'
    slug = 'p04_template_matching'
    description = 'Normalized cross-correlation against digit templates'
    is_online = False
    is_trainable = False

    def __init__(self, config=None):
        super().__init__(config)
        self.templates = {}

    def load(self):
        super().load()
        tmpl_h = self.config.get('template_height', 80)
        tmpl_w = self.config.get('template_width', 50)
        for d in range(10):
            self.templates[d] = generate_digit_template(d, tmpl_h, tmpl_w)

    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        debug = {}

        cropped = crop_roi(image, roi)
        gray = to_grayscale(cropped)
        gray = resize_height(gray, 100)

        # Threshold for digit isolation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = invert_if_dark(binary)

        num_digits = self.config.get('num_digits', 0)
        digit_imgs = isolate_digits(binary, num_digits)

        debug['binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        predicted = ''
        total_score = 0.0
        for digit_img in digit_imgs:
            best_digit = '?'
            best_score = -1
            h, w = digit_img.shape[:2]
            if h < 5 or w < 3:
                predicted += '?'
                continue

            for d, tmpl in self.templates.items():
                # Resize template to match digit size
                resized_tmpl = cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_LINEAR)
                # Normalized cross-correlation
                result = cv2.matchTemplate(
                    digit_img, resized_tmpl, cv2.TM_CCORR_NORMED
                )
                score = result.max()
                if score > best_score:
                    best_score = score
                    best_digit = str(d)

            predicted += best_digit
            total_score += best_score

        confidence = total_score / len(digit_imgs) if digit_imgs else 0.0

        return PipelineResult(
            predicted=predicted,
            confidence=confidence,
            debug_images=debug,
        )
