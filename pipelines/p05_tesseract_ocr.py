"""Pipeline 5: Tesseract OCR with digit whitelist."""

import cv2
import numpy as np
from pipelines import register
from pipelines.base import BasePipeline, PipelineResult, ROI
from pipelines.preprocessing import crop_roi, to_grayscale, resize_height, invert_if_dark, apply_morphology


@register
class TesseractOCRPipeline(BasePipeline):
    name = 'Tesseract OCR'
    slug = 'p05_tesseract_ocr'
    description = 'Tesseract with --psm 7 digit whitelist'
    is_online = False
    is_trainable = False

    def __init__(self, config=None):
        super().__init__(config)
        self._pytesseract = None

    def load(self):
        super().load()
        import pytesseract
        self._pytesseract = pytesseract

    def unload(self):
        self._pytesseract = None
        super().unload()

    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        if self._pytesseract is None:
            return PipelineResult(error='Tesseract not loaded')

        debug = {}

        cropped = crop_roi(image, roi)
        gray = to_grayscale(cropped)

        # Scale up for better OCR accuracy
        target_h = self.config.get('target_height', 200)
        gray = resize_height(gray, target_h)

        # CLAHE + threshold for better OCR input
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = invert_if_dark(binary)
        binary = apply_morphology(binary, 'close', 2)

        # Add white border for Tesseract
        border = 10
        padded = cv2.copyMakeBorder(binary, border, border, border, border,
                                     cv2.BORDER_CONSTANT, value=0)
        debug['ocr_input'] = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

        # Run Tesseract with digit-only whitelist
        # PSM 7 = treat as single text line
        # PSM 13 = raw line
        psm = self.config.get('psm', 7)
        tessdata = self.config.get('tessdata', None)
        lang = self.config.get('lang', 'eng')

        custom_config = f'--psm {psm} -c tessedit_char_whitelist=0123456789'
        if tessdata:
            custom_config += f' --tessdata-dir {tessdata}'

        try:
            text = self._pytesseract.image_to_string(padded, lang=lang, config=custom_config)
            # Clean output
            predicted = ''.join(c for c in text.strip() if c.isdigit())

            # Get confidence from detailed data
            data = self._pytesseract.image_to_data(padded, lang=lang, config=custom_config,
                                                    output_type=self._pytesseract.Output.DICT)
            confs = [int(c) for c in data['conf'] if int(c) > 0]
            confidence = sum(confs) / len(confs) / 100.0 if confs else 0.0

        except Exception as e:
            return PipelineResult(error=f'Tesseract error: {e}', debug_images=debug)

        return PipelineResult(
            predicted=predicted,
            confidence=confidence,
            debug_images=debug,
        )
