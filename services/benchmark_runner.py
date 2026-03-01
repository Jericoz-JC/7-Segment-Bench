"""Threaded benchmark orchestration with SSE event streaming."""

import json
import threading
import queue
import time
import cv2
from flask import current_app
from models import db
from models.benchmark import BenchmarkRun, BenchmarkResult
from models.image import Image
from models.label import Label
from services.metrics import compare_char_accuracy
from services.image_service import get_image_path
from services.label_service import benchmark_target, normalize_and_dedupe_dataset_labels
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

            # Phase 1: Initializing
            self._emit({'type': 'phase_change', 'phase': 'initializing'})

            selection = run.pipeline_configs.get('__selection', {})
            selected_image_ids = []
            if isinstance(selection, dict):
                raw_ids = selection.get('image_ids', [])
                if isinstance(raw_ids, list):
                    for item in raw_ids:
                        try:
                            selected_image_ids.append(int(item))
                        except Exception:
                            pass

            normalize_and_dedupe_dataset_labels(run.dataset_id)

            # Phase 2: Loading data
            self._emit({'type': 'phase_change', 'phase': 'loading_data'})

            # Get labeled images from dataset
            query = (
                db.session.query(Image, Label)
                .join(Label)
                .filter(Image.dataset_id == run.dataset_id)
            )
            if selected_image_ids:
                query = query.filter(Image.id.in_(selected_image_ids))

            labeled_images = query.all()
            if selected_image_ids:
                order = {img_id: i for i, img_id in enumerate(selected_image_ids)}
                labeled_images.sort(key=lambda row: order.get(row[0].id, len(order)))

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

            # ETA tracking
            bench_start_time = time.time()
            eta_counter = 0

            # Run each pipeline sequentially (one at a time for memory)
            for slug in slugs:
                # Phase 3: Pipeline
                self._emit({'type': 'phase_change', 'phase': f'pipeline:{slug}'})
                self._emit({'type': 'pipeline_start', 'pipeline': slug})

                pipeline_processed = 0
                pipeline_total = len(labeled_images)

                try:
                    pipe = pipelines.get_pipeline(slug, run.pipeline_configs.get(slug))
                    pipe.load()
                except Exception as e:
                    # Record errors for all images
                    for img, label in labeled_images:
                        gt_target = benchmark_target(label.ground_truth)
                        result = BenchmarkResult(
                            run_id=run.id, image_id=img.id, label_id=label.id,
                            pipeline_slug=slug, ground_truth=label.ground_truth,
                            error_message=f'Load failed: {e}',
                            char_total=len(gt_target),
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
                        pred_target = benchmark_target(pred_result.predicted)
                        gt_target = benchmark_target(label.ground_truth)
                        char_correct, char_total = compare_char_accuracy(
                            pred_target, gt_target
                        )

                        br = BenchmarkResult(
                            run_id=run.id, image_id=img.id, label_id=label.id,
                            pipeline_slug=slug,
                            predicted=pred_result.predicted,
                            ground_truth=label.ground_truth,
                            is_correct=(pred_target == gt_target),
                            char_correct=char_correct,
                            char_total=char_total,
                            latency_ms=pred_result.latency_ms,
                            confidence=pred_result.confidence,
                            error_message=pred_result.error,
                        )
                        db.session.add(br)

                    except Exception as e:
                        gt_target = benchmark_target(label.ground_truth)
                        br = BenchmarkResult(
                            run_id=run.id, image_id=img.id, label_id=label.id,
                            pipeline_slug=slug, ground_truth=label.ground_truth,
                            error_message=str(e),
                            char_total=len(gt_target),
                        )
                        db.session.add(br)

                    run.processed += 1
                    pipeline_processed += 1
                    db.session.commit()

                    self._emit({
                        'type': 'progress',
                        'processed': run.processed,
                        'total': total,
                        'pipeline': slug,
                        'image': img.filename,
                        'predicted': br.predicted,
                        'ground_truth': br.ground_truth,
                        'benchmark_target': benchmark_target(br.ground_truth),
                        'correct': br.is_correct,
                    })

                    # Per-pipeline progress
                    self._emit({
                        'type': 'pipeline_progress',
                        'pipeline': slug,
                        'pipeline_processed': pipeline_processed,
                        'pipeline_total': pipeline_total,
                    })

                    # ETA calculation every 5 images
                    eta_counter += 1
                    if eta_counter % 5 == 0 and run.processed > 0:
                        elapsed = time.time() - bench_start_time
                        remaining = total - run.processed
                        seconds_remaining = (elapsed / run.processed) * remaining
                        self._emit({
                            'type': 'eta',
                            'seconds_remaining': round(seconds_remaining, 1),
                        })

                try:
                    pipe.unload()
                except Exception:
                    pass

                self._emit({'type': 'pipeline_done', 'pipeline': slug})

            # Phase 4: Computing metrics
            self._emit({'type': 'phase_change', 'phase': 'computing_metrics'})

            from services.metrics import compute_run_metrics

            # Phase 5: Finalizing
            run.status = 'completed'
            run.summary = compute_run_metrics(run.id)
            db.session.commit()

            self._emit({'type': 'phase_change', 'phase': 'finalizing'})
            self._emit({'type': 'completed', 'run_id': run.id})
            self._queue.put(None)
