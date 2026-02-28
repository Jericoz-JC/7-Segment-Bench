"""Unit tests for CV pipelines against synthetic 7-segment images."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.synth import render_display, render_digit, add_noise
from pipelines.base import ROI
from pipelines.segment_map import (
    extract_segment_values, classify_digit, isolate_digits, read_display,
    DIGIT_SEGMENTS
)


class TestSegmentMap:
    """Test segment geometry and digit classification."""

    def test_classify_all_digits(self):
        """Each digit 0-9 should be correctly classified from a clean render."""
        import cv2
        for digit in range(10):
            img = render_digit(digit, 120, 70, fg_color=(255, 255, 255), bg_color=(0, 0, 0))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            seg_vals = extract_segment_values(binary)
            predicted, conf = classify_digit(seg_vals)
            assert predicted == str(digit), f'Digit {digit}: predicted {predicted}'
            assert conf > 0.5

    def test_isolate_digits_contour(self):
        """Contour-based isolation should find the right number of digits."""
        import cv2
        img = render_display('1234', gap=15)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        digits = isolate_digits(binary, num_digits=4)
        assert len(digits) == 4

    def test_read_display_clean(self):
        """Full pipeline on clean synthetic image."""
        import cv2
        for gt in ['0123', '4567', '8900']:
            img = render_display(gt)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            predicted, conf = read_display(binary, num_digits=len(gt))
            assert predicted == gt, f'Expected {gt}, got {predicted}'


class TestPipeline01GlobalThreshold:
    """Test Otsu threshold pipeline."""

    def test_clean_image(self):
        from pipelines.p01_global_threshold import GlobalThresholdPipeline
        pipe = GlobalThresholdPipeline({'num_digits': 4})
        pipe.load()
        img = render_display('0237')
        roi = ROI.full_image(img)
        result = pipe.predict_timed(img, roi)
        pipe.unload()
        assert result.predicted == '0237', f'Got: {result.predicted}'
        assert result.latency_ms > 0

    def test_noisy_image(self):
        from pipelines.p01_global_threshold import GlobalThresholdPipeline
        pipe = GlobalThresholdPipeline({'num_digits': 4})
        pipe.load()
        img = add_noise(render_display('1234'), 15)
        roi = ROI.full_image(img)
        result = pipe.predict(img, roi)
        pipe.unload()
        # With moderate noise, should still be close
        assert len(result.predicted) == 4


class TestPipeline02AdaptiveCLAHE:
    """Test CLAHE + adaptive threshold pipeline."""

    def test_clean_image(self):
        from pipelines.p02_adaptive_clahe import AdaptiveCLAHEPipeline
        pipe = AdaptiveCLAHEPipeline({'num_digits': 4})
        pipe.load()
        img = render_display('5678')
        roi = ROI.full_image(img)
        result = pipe.predict(img, roi)
        pipe.unload()
        assert result.predicted == '5678', f'Got: {result.predicted}'


class TestPipeline03HSVColor:
    """Test HSV color filter pipeline."""

    def test_green_led(self):
        from pipelines.p03_hsv_color import HSVColorPipeline
        pipe = HSVColorPipeline({'color': 'green', 'num_digits': 4})
        pipe.load()
        # Green LED display
        img = render_display('9012', fg_color=(0, 255, 0))
        roi = ROI.full_image(img)
        result = pipe.predict(img, roi)
        pipe.unload()
        assert result.predicted == '9012', f'Got: {result.predicted}'

    def test_red_led(self):
        from pipelines.p03_hsv_color import HSVColorPipeline
        pipe = HSVColorPipeline({'color': 'red', 'num_digits': 4})
        pipe.load()
        img = render_display('4321', fg_color=(0, 0, 255))
        roi = ROI.full_image(img)
        result = pipe.predict(img, roi)
        pipe.unload()
        assert result.predicted == '4321', f'Got: {result.predicted}'


class TestPipeline04TemplateMatching:
    """Test template matching pipeline."""

    def test_clean_image(self):
        from pipelines.p04_template_matching import TemplateMatchingPipeline
        pipe = TemplateMatchingPipeline({'num_digits': 4})
        pipe.load()
        img = render_display('8642')
        roi = ROI.full_image(img)
        result = pipe.predict(img, roi)
        pipe.unload()
        assert result.predicted == '8642', f'Got: {result.predicted}'


class TestMetrics:
    """Test metric computation utilities."""

    def test_char_accuracy(self):
        from services.metrics import compare_char_accuracy
        correct, total = compare_char_accuracy('1234', '1234')
        assert correct == 4
        assert total == 4

    def test_char_accuracy_partial(self):
        from services.metrics import compare_char_accuracy
        correct, total = compare_char_accuracy('1234', '1235')
        assert correct == 3
        assert total == 4

    def test_char_accuracy_length_mismatch(self):
        from services.metrics import compare_char_accuracy
        correct, total = compare_char_accuracy('123', '1234')
        assert correct == 3
        assert total == 4

    def test_char_accuracy_empty(self):
        from services.metrics import compare_char_accuracy
        correct, total = compare_char_accuracy('', '')
        assert correct == 0
        assert total == 0


class TestROI:
    """Test ROI data class."""

    def test_crop(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        roi = ROI(x=10, y=20, width=50, height=30)
        cropped = roi.crop(img)
        assert cropped.shape == (30, 50, 3)

    def test_full_image(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        roi = ROI.full_image(img)
        assert roi.x == 0 and roi.y == 0
        assert roi.width == 200 and roi.height == 100

    def test_from_dict(self):
        roi = ROI.from_dict({'x': 5, 'y': 10, 'width': 100, 'height': 50})
        assert roi.x == 5 and roi.width == 100
