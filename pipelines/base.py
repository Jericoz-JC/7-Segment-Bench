"""Abstract base pipeline and shared data classes for all CV pipelines."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ROI:
    """Region of interest within an image."""
    x: int
    y: int
    width: int
    height: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop image to this ROI. Image is (H, W) or (H, W, C)."""
        y2 = self.y + self.height
        x2 = self.x + self.width
        return image[self.y:y2, self.x:x2]

    @classmethod
    def from_dict(cls, d: dict) -> 'ROI':
        return cls(x=d['x'], y=d['y'], width=d['width'], height=d['height'])

    @classmethod
    def full_image(cls, image: np.ndarray) -> 'ROI':
        h, w = image.shape[:2]
        return cls(x=0, y=0, width=w, height=h)


@dataclass
class PipelineResult:
    """Result from a pipeline prediction."""
    predicted: str = ''
    confidence: float = 0.0
    latency_ms: float = 0.0
    debug_images: dict = field(default_factory=dict)  # name -> np.ndarray
    error: str = ''

    @property
    def success(self) -> bool:
        return self.error == '' and self.predicted != ''


class BasePipeline(ABC):
    """Abstract base class for all CV pipelines."""

    name: str = 'base'
    slug: str = 'base'
    description: str = ''
    is_online: bool = False
    is_trainable: bool = False

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._loaded = False

    def load(self):
        """Load model/resources into memory. Called before prediction batch."""
        self._loaded = True

    def unload(self):
        """Free model/resources. Called after prediction batch."""
        self._loaded = False

    @abstractmethod
    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        """Run prediction on a single image with given ROI.

        Args:
            image: BGR image as numpy array (as loaded by cv2.imread)
            roi: Region of interest containing the display digits

        Returns:
            PipelineResult with predicted digits, confidence, debug images
        """
        ...

    def predict_timed(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        """Wrapper that measures prediction latency."""
        start = time.perf_counter()
        try:
            result = self.predict(image, roi)
        except Exception as e:
            result = PipelineResult(error=str(e))
        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed
        return result

    def __repr__(self):
        return f'<{self.__class__.__name__} slug={self.slug}>'
