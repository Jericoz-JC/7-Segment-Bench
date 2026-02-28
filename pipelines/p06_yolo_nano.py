"""Pipeline 6: YOLOv8n detection — each digit 0-9 as a class.

Uses NCNN export on Pi (ARM64), PyTorch on Windows/x86.
"""

import platform
import cv2
import numpy as np
from pipelines import register
from pipelines.base import BasePipeline, PipelineResult, ROI
from pipelines.preprocessing import crop_roi


@register
class YOLONanoPipeline(BasePipeline):
    name = 'YOLOv8 Nano'
    slug = 'p06_yolo_nano'
    description = 'Object detection with YOLOv8n, digits 0-9 as classes'
    is_online = False
    is_trainable = True

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None

    def load(self):
        super().load()
        from ultralytics import YOLO

        model_path = self.config.get('model_path', '')
        if not model_path:
            # Use pretrained yolov8n as fallback (won't know digits,
            # but allows pipeline to run)
            model_path = 'yolov8n.pt'

        # On Pi, try NCNN format
        is_arm = platform.machine().startswith(('aarch64', 'arm'))
        ncnn_path = self.config.get('ncnn_path', '')
        if is_arm and ncnn_path:
            model_path = ncnn_path

        self._model = YOLO(model_path)

    def unload(self):
        self._model = None
        super().unload()

    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        if self._model is None:
            return PipelineResult(error='YOLO model not loaded')

        debug = {}
        cropped = crop_roi(image, roi)
        debug['input'] = cropped.copy()

        try:
            results = self._model(cropped, verbose=False)
        except Exception as e:
            return PipelineResult(error=f'YOLO inference error: {e}', debug_images=debug)

        if not results or len(results) == 0:
            return PipelineResult(predicted='', confidence=0.0, debug_images=debug)

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return PipelineResult(predicted='', confidence=0.0, debug_images=debug)

        # Sort detections left-to-right by x coordinate
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            cx = (x1 + x2) / 2
            detections.append((cx, cls_id, conf, (x1, y1, x2, y2)))

        detections.sort(key=lambda d: d[0])

        predicted = ''.join(str(d[1]) for d in detections)
        avg_conf = sum(d[2] for d in detections) / len(detections)

        # Draw debug image
        annotated = cropped.copy()
        for cx, cls_id, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f'{cls_id}:{conf:.2f}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        debug['detections'] = annotated

        return PipelineResult(
            predicted=predicted,
            confidence=avg_conf,
            debug_images=debug,
        )
