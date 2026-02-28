"""Download and import curated external datasets into the app database."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

import requests
import yaml
from PIL import Image as PILImage
from werkzeug.datastructures import FileStorage

from models import db
from models.dataset import Dataset
from services.external_dataset_catalog import (
    BASE_DIR,
    CuratedDataset,
    check_access,
    get_curated_dataset,
)
from services.image_service import save_uploaded_image
from services.label_service import import_labels_json


EXTERNAL_DIR = BASE_DIR / "external_datasets"
STAGING_DIR = EXTERNAL_DIR / "staging"

HF_DATASET_ID = "HelloImMrGrey/7SEG_OCR"
MENDELEY_ID = "fnn44p4mj8"

ProgressCallback = Callable[[str, int, int, str], None]


class ExternalDatasetInstallError(RuntimeError):
    """Raised when dataset install fails."""


class ExternalDatasetValidationError(ValueError):
    """Raised when dataset install request is invalid."""


class ProgressReporter:
    def __init__(self, callback: ProgressCallback | None):
        self._callback = callback

    def emit(self, phase: str, processed: int, total: int, message: str) -> None:
        if self._callback:
            self._callback(phase, processed, total, message)


class ImportSample:
    def __init__(
        self,
        image_path: Path,
        filename: str,
        ground_truth: str,
        roi_x: int,
        roi_y: int,
        roi_width: int,
        roi_height: int,
        display_type: str = "led",
    ):
        self.image_path = image_path
        self.filename = filename
        self.ground_truth = ground_truth
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.display_type = display_type


def validate_install_request(dataset_key: str, mode: str) -> CuratedDataset:
    if mode not in {"quick", "full"}:
        raise ExternalDatasetValidationError("mode must be either 'quick' or 'full'.")

    try:
        item = get_curated_dataset(dataset_key)
    except KeyError as exc:
        msg = exc.args[0] if exc.args else str(exc)
        raise ExternalDatasetValidationError(msg) from exc

    if mode == "full" and not item.supports_full:
        raise ExternalDatasetValidationError("Full install is not supported for this dataset.")

    allowed, reason = check_access(item)
    if not allowed:
        raise ExternalDatasetValidationError(reason)

    return item


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_label(text: str) -> str:
    clean = re.sub(r"\s+", "", str(text or "").strip())
    return clean[:50]


def guess_symbol(name: str) -> str:
    s = str(name).strip()
    if not s:
        return ""
    if len(s) == 1:
        return s
    m = re.search(r"([0-9])", s)
    if m:
        return m.group(1)
    s_low = s.lower()
    if "minus" in s_low:
        return "-"
    if "dot" in s_low or "decimal" in s_low:
        return "."
    return re.sub(r"\s+", "", s)[:1]


def read_image_size(path: Path) -> tuple[int, int]:
    with PILImage.open(path) as im:
        return im.size


def download_file(url: str, output_path: Path, retries: int = 4) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            return
        except Exception as exc:
            last_err = exc
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(1.2 * attempt)
    if last_err is not None:
        raise last_err


def mendeley_download_assets(progress: ProgressReporter | None = None) -> tuple[Path, Path]:
    target = EXTERNAL_DIR / "mendeley_fnn44p4mj8_v1"
    target.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}", timeout=60).json()
    ann_path = None
    zip_path = None

    if progress:
        progress.emit("download", 0, len(meta.get("files", [])), "Fetching Mendeley files")

    for idx, item in enumerate(meta.get("files", []), start=1):
        filename = item["filename"]
        url = item["content_details"]["download_url"]
        out = target / filename
        download_file(url, out)
        if filename.lower().endswith(".json"):
            ann_path = out
        if filename.lower().endswith(".zip"):
            zip_path = out
        if progress:
            progress.emit("download", idx, len(meta.get("files", [])), f"Downloaded {filename}")

    if ann_path is None:
        raise ExternalDatasetInstallError("Could not find Mendeley annotation JSON file.")
    if zip_path is None:
        raise ExternalDatasetInstallError("Could not find Mendeley raw image ZIP file.")
    return ann_path, zip_path


def build_mendeley_samples(max_images: int | None = None, progress: ProgressReporter | None = None) -> list[ImportSample]:
    ann_path, _ = mendeley_download_assets(progress=progress)
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    category_to_symbol: dict[int, str] = {}
    for c in data.get("categories", []):
        category_to_symbol[int(c["id"])] = guess_symbol(c.get("name", ""))

    image_by_id = {img["id"]: img for img in data.get("images", [])}
    ann_by_image: dict[str, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        ann_by_image[ann["image_id"]].append(ann)

    out_dir = STAGING_DIR / "mendeley_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[ImportSample] = []
    image_ids = list(ann_by_image.keys())
    for idx, image_id in enumerate(image_ids, start=1):
        img_meta = image_by_id.get(image_id)
        if not img_meta:
            continue
        url = img_meta.get("file_name", "")
        if not url:
            continue

        parsed = Path(unquote(urlparse(url).path))
        base_name = parsed.name or f"{image_id}.jpg"
        filename = f"{image_id}_{base_name}"
        image_path = out_dir / filename
        try:
            download_file(url, image_path, retries=5)
        except Exception:
            continue

        x_min = 10**9
        y_min = 10**9
        x_max = 0.0
        y_max = 0.0
        symbols: list[tuple[float, str]] = []
        for ann in ann_by_image[image_id]:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            sym = category_to_symbol.get(int(ann.get("category_id", -1)), "")
            symbols.append((x + w / 2.0, sym))
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        symbols.sort(key=lambda t: t[0])
        gt = sanitize_label("".join(s for _, s in symbols))
        if not gt:
            continue

        width, height = read_image_size(image_path)
        roi_x = max(0, int(x_min))
        roi_y = max(0, int(y_min))
        roi_w = max(1, min(width - roi_x, int(round(x_max - x_min))))
        roi_h = max(1, min(height - roi_y, int(round(y_max - y_min))))

        samples.append(
            ImportSample(
                image_path=image_path,
                filename=filename,
                ground_truth=gt,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_width=roi_w,
                roi_height=roi_h,
                display_type="lcd",
            )
        )

        if progress:
            progress.emit("prepare", idx, len(image_ids), f"Prepared {len(samples)} Mendeley samples")

        if max_images is not None and len(samples) >= max_images:
            break

    return samples


def build_hf_samples(max_images: int | None = None, progress: ProgressReporter | None = None) -> list[ImportSample]:
    from datasets import load_dataset

    out_dir = STAGING_DIR / "hf_7seg_ocr_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_DATASET_ID, split="train")
    total = len(ds)

    samples: list[ImportSample] = []
    for idx, row in enumerate(ds):
        gt = sanitize_label(row.get("text", ""))
        if not gt:
            continue
        filename = f"hf7seg_{idx:05d}.png"
        image_path = out_dir / filename
        if not image_path.exists():
            img = row["image"]
            img.save(image_path)
        width, height = read_image_size(image_path)
        samples.append(
            ImportSample(
                image_path=image_path,
                filename=filename,
                ground_truth=gt,
                roi_x=0,
                roi_y=0,
                roi_width=width,
                roi_height=height,
                display_type="led",
            )
        )

        if progress and idx % 25 == 0:
            progress.emit("prepare", idx + 1, total, f"Prepared {len(samples)} Hugging Face samples")

        if max_images is not None and len(samples) >= max_images:
            break

    return samples


def parse_roboflow_yolo_export(dataset_root: Path, max_images: int | None = None) -> list[ImportSample]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise ExternalDatasetInstallError(f"Missing data.yaml in {dataset_root}")
    with open(data_yaml, "r", encoding="utf-8") as fh:
        ycfg = yaml.safe_load(fh)

    names = ycfg.get("names", [])
    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    else:
        id_to_name = {i: str(v) for i, v in enumerate(names)}

    samples: list[ImportSample] = []
    for split in ["train", "valid", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for image_path in sorted(img_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            label_path = lbl_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                continue

            width, height = read_image_size(image_path)
            detections = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                xc = float(parts[1]) * width
                yc = float(parts[2]) * height
                bw = float(parts[3]) * width
                bh = float(parts[4]) * height
                x1 = max(0.0, xc - bw / 2.0)
                y1 = max(0.0, yc - bh / 2.0)
                x2 = min(float(width), xc + bw / 2.0)
                y2 = min(float(height), yc + bh / 2.0)
                sym = guess_symbol(id_to_name.get(cls_id, str(cls_id)))
                detections.append((xc, sym, x1, y1, x2, y2))
            if not detections:
                continue

            detections.sort(key=lambda d: d[0])
            gt = sanitize_label("".join(d[1] for d in detections))
            if not gt:
                continue

            x_min = int(max(0.0, min(d[2] for d in detections)))
            y_min = int(max(0.0, min(d[3] for d in detections)))
            x_max = int(min(float(width), max(d[4] for d in detections)))
            y_max = int(min(float(height), max(d[5] for d in detections)))
            roi_w = max(1, x_max - x_min)
            roi_h = max(1, y_max - y_min)

            samples.append(
                ImportSample(
                    image_path=image_path,
                    filename=f"{split}_{image_path.name}",
                    ground_truth=gt,
                    roi_x=x_min,
                    roi_y=y_min,
                    roi_width=roi_w,
                    roi_height=roi_h,
                    display_type="led",
                )
            )
            if max_images is not None and len(samples) >= max_images:
                return samples

    return samples


def build_roboflow_samples(
    workspace: str,
    project_slug: str,
    local_name: str,
    max_images: int | None = None,
    progress: ProgressReporter | None = None,
) -> list[ImportSample]:
    import os
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise ExternalDatasetInstallError("ROBOFLOW_API_KEY is required for Roboflow datasets.")

    out_dir = EXTERNAL_DIR / local_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    versions = project.versions()
    if not versions:
        raise ExternalDatasetInstallError(f"No versions found for {workspace}/{project_slug}")
    latest_version = max(int(v.version) for v in versions)
    if progress:
        progress.emit("download", 0, 1, f"Downloading {workspace}/{project_slug} v{latest_version}")
    dataset = project.version(latest_version).download("yolov8", location=str(out_dir))
    dataset_root = Path(dataset.location)
    return parse_roboflow_yolo_export(dataset_root, max_images=max_images)


def ensure_unique_dataset_name(base_name: str) -> str:
    return f"{base_name}_{now_tag()}_{uuid.uuid4().hex[:6]}"


def import_samples_into_app(
    item: CuratedDataset,
    mode: str,
    samples: list[ImportSample],
    progress: ProgressReporter | None = None,
) -> dict:
    if not samples:
        raise ExternalDatasetInstallError("No samples were prepared for import.")

    dataset_name = ensure_unique_dataset_name(f"{item.dataset_name_prefix}_{mode}")
    dataset = Dataset(
        name=dataset_name,
        description=f"Imported from {item.name} ({mode})",
        lighting_tag="mixed",
    )
    db.session.add(dataset)
    db.session.commit()

    uploaded_total = 0
    upload_errors = 0
    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        try:
            with open(sample.image_path, "rb") as fh:
                storage = FileStorage(stream=fh, filename=sample.filename)
                save_uploaded_image(storage, dataset)
            uploaded_total += 1
        except Exception:
            upload_errors += 1

        if progress and (idx % 25 == 0 or idx == total):
            progress.emit("import", idx, total, f"Imported {idx}/{total} images")

    labels_payload = [
        {
            "filename": s.filename,
            "ground_truth": s.ground_truth,
            "roi_x": s.roi_x,
            "roi_y": s.roi_y,
            "roi_width": s.roi_width,
            "roi_height": s.roi_height,
            "display_type": s.display_type,
        }
        for s in samples
    ]
    imported_labels = import_labels_json(dataset.id, json.dumps(labels_payload))

    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "uploaded_images": uploaded_total,
        "upload_errors": upload_errors,
        "labels_imported": imported_labels,
    }


def install_external_dataset(
    dataset_key: str,
    mode: str,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    item = validate_install_request(dataset_key, mode)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(progress_cb)

    max_images = item.quick_count if mode == "quick" else None
    reporter.emit("prepare", 0, 1, f"Preparing {item.name}")

    if dataset_key == "hf_7seg_ocr":
        samples = build_hf_samples(max_images=max_images, progress=reporter)
    elif dataset_key == "mendeley_fnn44p4mj8":
        samples = build_mendeley_samples(max_images=max_images, progress=reporter)
    elif dataset_key == "roboflow_seven_segment_digits":
        samples = build_roboflow_samples(
            workspace="charlie-srmko",
            project_slug="seven-segment-digits-uptcy",
            local_name="roboflow_seven_segment_digits",
            max_images=max_images,
            progress=reporter,
        )
    elif dataset_key == "roboflow_seven_segment_display_ocr":
        samples = build_roboflow_samples(
            workspace="fyp-zodww",
            project_slug="seven-segment-display-ocr-lguqw",
            local_name="roboflow_seven_segment_display_ocr",
            max_images=max_images,
            progress=reporter,
        )
    else:
        raise ExternalDatasetInstallError(f"Unsupported dataset key: {dataset_key}")

    reporter.emit("import", 0, len(samples), f"Importing {len(samples)} samples")
    result = import_samples_into_app(item=item, mode=mode, samples=samples, progress=reporter)
    reporter.emit("done", len(samples), len(samples), "Install complete")

    return {
        "dataset_key": dataset_key,
        "mode": mode,
        "sample_count": len(samples),
        **result,
    }
