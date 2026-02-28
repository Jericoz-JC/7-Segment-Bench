"""Threaded benchmark orchestration with SSE event streaming."""

import json
import threading
import queue
import cv2
from flask import current_app
from models import db
from models.benchmark import BenchmarkRun, BenchmarkResult
from models.image import Image
from models.label import Label
from services.metrics import compare_char_accuracy
from services.image_service import get_image_path
import pipelines
from pipelines.base import ROI


class BenchmarkRunner:
    """Runs benchmark in a background thread, produces SSE events."""

    def __init__(self, run_id: int):
        self.run_id = run_id
        self._queue = queue.Queue()
        self._thread = None
        self._app = current_app._get_current_object()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def events(self):
        """Yield SSE event strings."""
        while True:
            try:
                event = self._queue.get(timeout=30)
                if event is None:
                    break
                yield json.dumps(event)
            except queue.Empty:
                # Send keepalive
                yield json.dumps({'type': 'keepalive'})

    def _emit(self, data: dict):
        self._queue.put(data)

    def _run(self):
        with self._app.app_context():
            run = BenchmarkRun.query.get(self.run_id)
            if not run:
                self._emit({'type': 'error', 'message': 'Run not found'})
                self._queue.put(None)
                return

            # Get labeled images from dataset
            labeled_images = (
                db.session.query(Image, Label)
                .join(Label)
                .filter(Image.dataset_id == run.dataset_id)
                .all()
            )

            if not labeled_images:
                run.status = 'failed'
                run.summary = {'error': 'No labeled images in dataset'}
                db.session.commit()
                self._emit({'type': 'error', 'message': 'No labeled images'})
                self._queue.put(None)
                return

            slugs = run.pipeline_slugs
            total = len(labeled_images) * len(slugs)
            run.total = total
            run.processed = 0
            run.status = 'running'
            db.session.commit()

            self._emit({
                'type': 'started',
                'total': total,
                'pipelines': slugs,
                'images': len(labeled_images),
            })

            # Run each pipeline sequentially (one at a time for memory)
            for slug in slugs:
                self._emit({'type': 'pipeline_start', 'pipeline': slug})

                try:
                    pipe = pipelines.get_pipeline(slug, run.pipeline_configs.get(slug))
                    pipe.load()
                except Exception as e:
                    # Record errors for all images
                    for img, label in labeled_images:
                        result = BenchmarkResult(
                            run_id=run.id, image_id=img.id, label_id=label.id,
                            pipeline_slug=slug, ground_truth=label.ground_truth,
                            error_message=f'Load failed: {e}',
                            char_total=len(label.ground_truth),
                        )
                        db.session.add(result)
                        run.processed += 1
                    db.session.commit()
                    self._emit({'type': 'pipeline_error', 'pipeline': slug, 'error': str(e)})
                    continue

                for img, label in labeled_images:
                    try:
                        image_data = cv2.imread(get_image_path(img))
                        if image_data is None:
                            raise ValueError(f'Could not read image: {img.filepath}')

                        roi = ROI(
                            x=label.roi_x, y=label.roi_y,
                            width=label.roi_width or img.width,
                            height=label.roi_height or img.height,
                        )

                        pred_result = pipe.predict_timed(image_data, roi)
                        char_correct, char_total = compare_char_accuracy(
                            pred_result.predicted, label.ground_truth
                        )

                        br = BenchmarkResult(
                            run_id=run.id, image_id=img.id, label_id=label.id,
                            pipeline_slug=slug,
                            predicted=pred_result.predicted,
                            ground_truth=label.ground_truth,
                            is_correct=(pred_result.predicted == label.ground_truth),
                            char_correct=char_correct,
                            char_total=char_total,
                            latency_ms=pred_result.latency_ms,
                            confidence=pred_result.confidence,
                            error_message=pred_result.error,
                        )
                        db.session.add(br)

                    except Exception as e:
                        br = BenchmarkResult(
                            run_id=run.id, image_id=img.id, label_id=label.id,
                            pipeline_slug=slug, ground_truth=label.ground_truth,
                            error_message=str(e),
                            char_total=len(label.ground_truth),
                        )
                        db.session.add(br)

                    run.processed += 1
                    db.session.commit()

                    self._emit({
                        'type': 'progress',
                        'processed': run.processed,
                        'total': total,
                        'pipeline': slug,
                        'image': img.filename,
                        'predicted': br.predicted,
                        'ground_truth': br.ground_truth,
                        'correct': br.is_correct,
                    })

                try:
                    pipe.unload()
                except Exception:
                    pass

                self._emit({'type': 'pipeline_done', 'pipeline': slug})

            # Finalize
            from services.metrics import compute_run_metrics
            run.status = 'completed'
            run.summary = compute_run_metrics(run.id)
            db.session.commit()

            self._emit({'type': 'completed', 'run_id': run.id})
            self._queue.put(None)
