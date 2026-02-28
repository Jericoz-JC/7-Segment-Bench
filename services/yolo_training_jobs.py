"""Background training manager for YOLO pipeline jobs."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask

from models.dataset import Dataset
from services.yolo_trainer import (
    build_split_yolo_dataset,
    default_yolo_training_root,
    make_training_run_name,
    register_trained_model,
    train_yolo_model,
)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class YoloTrainingJobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start_training(self, app: Flask, dataset_id: int, params: dict) -> str:
        job_id = uuid.uuid4().hex
        job = {
            'job_id': job_id,
            'dataset_id': dataset_id,
            'status': 'pending',
            'phase': 'queued',
            'message': 'Queued',
            'params': params,
            'result': None,
            'error': '',
            'created_at': _ts(),
            'updated_at': _ts(),
        }
        with self._lock:
            self._jobs[job_id] = job

        worker = threading.Thread(
            target=self._run,
            args=(app, job_id, dataset_id, params),
            daemon=True,
        )
        worker.start()
        return job_id

    def get_status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def _update(self, job_id: str, **fields):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            job['updated_at'] = _ts()

    def _run(self, app: Flask, job_id: str, dataset_id: int, params: dict):
        self._update(job_id, status='running', phase='prepare', message='Preparing training data')
        try:
            with app.app_context():
                dataset = Dataset.query.get(dataset_id)
                if not dataset:
                    raise RuntimeError(f'Dataset {dataset_id} not found')

                run_name = params.get('run_name') or make_training_run_name(dataset.name)
                training_root = Path(default_yolo_training_root())
                split_dir = training_root / 'splits' / run_name
                project_dir = training_root / 'runs'

                split_info = build_split_yolo_dataset(
                    dataset_id=dataset_id,
                    output_dir=str(split_dir),
                    val_ratio=float(params.get('val_ratio', 0.2)),
                    seed=int(params.get('seed', 42)),
                )

                self._update(job_id, phase='train', message='Training YOLO model')
                train_info = train_yolo_model(
                    data_yaml_path=split_info['yaml_path'],
                    project_dir=str(project_dir),
                    run_name=run_name,
                    epochs=int(params.get('epochs', 30)),
                    imgsz=int(params.get('imgsz', 640)),
                    batch=int(params.get('batch', 8)),
                    device=str(params.get('device', '')).strip() or None,
                )

                self._update(job_id, phase='register', message='Registering trained model')
                model = register_trained_model(
                    dataset_id=dataset_id,
                    model_path=train_info['best_model_path'],
                    run_name=run_name,
                    metrics=train_info.get('metrics', {}),
                    activate=bool(params.get('activate', True)),
                )

                result = {
                    'model_id': model.id,
                    'model_name': model.name,
                    'model_path': model.model_path,
                    'is_active': model.is_active,
                    'split': split_info,
                    'training': train_info,
                }
                self._update(
                    job_id,
                    status='completed',
                    phase='done',
                    message='Training completed',
                    result=result,
                    error='',
                )
        except Exception as exc:
            self._update(
                job_id,
                status='failed',
                phase='error',
                message='Training failed',
                error=str(exc),
            )
