"""Background job manager for curated external dataset installs."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone

from flask import Flask

from services.external_dataset_installer import install_external_dataset


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExternalDatasetInstallManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start_install(self, app: Flask, dataset_key: str, mode: str) -> str:
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "dataset_key": dataset_key,
            "mode": mode,
            "status": "pending",
            "phase": "queued",
            "processed": 0,
            "total": 0,
            "progress_pct": 0.0,
            "message": "Queued",
            "result": None,
            "error": "",
            "created_at": _ts(),
            "updated_at": _ts(),
        }
        with self._lock:
            self._jobs[job_id] = job

        worker = threading.Thread(
            target=self._run,
            args=(app, job_id, dataset_key, mode),
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
            if job["total"] > 0:
                job["progress_pct"] = round(100 * job["processed"] / job["total"], 1)
            else:
                job["progress_pct"] = 0.0
            job["updated_at"] = _ts()

    def _run(self, app: Flask, job_id: str, dataset_key: str, mode: str):
        self._update(
            job_id,
            status="running",
            phase="prepare",
            message="Preparing dataset",
        )

        def progress_cb(phase: str, processed: int, total: int, message: str):
            self._update(
                job_id,
                status="running",
                phase=phase,
                processed=processed,
                total=total,
                message=message,
            )

        try:
            with app.app_context():
                result = install_external_dataset(
                    dataset_key=dataset_key,
                    mode=mode,
                    progress_cb=progress_cb,
                )
            self._update(
                job_id,
                status="completed",
                phase="done",
                processed=result.get("sample_count", 0),
                total=result.get("sample_count", 0),
                progress_pct=100.0,
                message="Install completed",
                result=result,
                error="",
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                phase="error",
                message="Install failed",
                error=str(exc),
            )
