"""Focused validation tests for p05/p06/p07 pathways."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipelines.base import ROI


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBoxes:
    def __init__(self):
        # Intentionally out of x-order: class 7 at x=70, class 3 at x=20
        self.xyxy = [
            _FakeTensor([60, 10, 80, 40]),
            _FakeTensor([10, 8, 30, 38]),
        ]
        self.cls = [
            _FakeTensor([7]),
            _FakeTensor([3]),
        ]
        self.conf = [
            _FakeTensor([0.8]),
            _FakeTensor([0.9]),
        ]

    def __len__(self):
        return len(self.xyxy)


def test_p05_whitelist_strips_mixed_symbols():
    from pipelines.p05_tesseract_ocr import TesseractOCRPipeline

    class FakeOutput:
        DICT = object()

    class FakeTesseract:
        Output = FakeOutput()

        @staticmethod
        def image_to_string(*args, **kwargs):
            return '12A-3'

        @staticmethod
        def image_to_data(*args, **kwargs):
            return {'conf': ['90', '80']}

    pipe = TesseractOCRPipeline({'psm': 7})
    pipe._pytesseract = FakeTesseract()
    image = np.zeros((60, 180, 3), dtype=np.uint8)
    result = pipe.predict(image, ROI.full_image(image))
    assert result.error == ''
    assert result.predicted == '123'
    assert result.confidence > 0


def test_p06_left_to_right_sorting():
    from pipelines.p06_yolo_nano import _decode_detections

    detections = _decode_detections(_FakeBoxes())
    predicted = ''.join(str(d[1]) for d in detections)
    assert predicted == '37'


def test_p07_openrouter_load_uses_openai_compatible_client(monkeypatch):
    from pipelines.p07_llm_vision import LLMVisionPipeline

    captured = {}

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeOpenAIModule:
        OpenAI = FakeOpenAIClient

    monkeypatch.setitem(sys.modules, 'openai', FakeOpenAIModule())

    pipe = LLMVisionPipeline({'provider': 'openrouter', 'api_key': 'or-key'})
    pipe.load()
    assert pipe._client is not None
    assert captured['base_url'] == 'https://openrouter.ai/api/v1'
    assert captured['api_key'] == 'or-key'


def test_p07_ollama_load_allows_empty_key(monkeypatch):
    from pipelines.p07_llm_vision import LLMVisionPipeline

    captured = {}

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeOpenAIModule:
        OpenAI = FakeOpenAIClient

    monkeypatch.setitem(sys.modules, 'openai', FakeOpenAIModule())

    pipe = LLMVisionPipeline({'provider': 'ollama', 'api_key': ''})
    pipe.load()
    assert pipe._client is not None
    assert captured['base_url'] == 'http://localhost:11434/v1'
    assert captured['api_key'] == 'ollama-local'
