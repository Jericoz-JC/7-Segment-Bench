"""Download external 7-segment datasets, import into app, and run benchmark.

Usage:
    ..\\venv\\Scripts\\python.exe scripts\\install_and_run_external_datasets.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import requests
import yaml
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from models.benchmark import BenchmarkRun  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = BASE_DIR / "external_datasets"
STAGING_DIR = EXTERNAL_DIR / "staging"

HF_DATASET_ID = "HelloImMrGrey/7SEG_OCR"
MENDELEY_ID = "fnn44p4mj8"


@dataclass
class ImportSample:
    image_path: Path
    filename: str
    ground_truth: str
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    display_type: str = "led"


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
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(1.5 * attempt)
    if last_err is not None:
        raise last_err


def mendeley_download_assets() -> tuple[Path, Path]:
    target = EXTERNAL_DIR / "mendeley_fnn44p4mj8_v1"
    target.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}", timeout=60).json()
    ann_path = None
    zip_path = None
    for f in meta["files"]:
        filename = f["filename"]
        url = f["content_details"]["download_url"]
        out = target / filename
        download_file(url, out)
        if filename.lower().endswith(".json"):
            ann_path = out
        if filename.lower().endswith(".zip"):
            zip_path = out
    if ann_path is None:
        raise RuntimeError("Could not find Mendeley annotation JSON file.")
    if zip_path is None:
        raise RuntimeError("Could not find Mendeley raw image ZIP file.")
    return ann_path, zip_path


def build_mendeley_samples(max_images: int | None = None) -> list[ImportSample]:
    ann_path, _ = mendeley_download_assets()
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    category_to_symbol: dict[int, str] = {}
    for c in data.get("categories", []):
        category_to_symbol[int(c["id"])] = guess_symbol(c.get("name", ""))

    image_by_id = {img["id"]: img for img in data.get("images", [])}
    ann_by_image: dict[str, list[dict]] = defaultdict(list)
    for a in data.get("annotations", []):
        ann_by_image[a["image_id"]].append(a)

    out_dir = STAGING_DIR / "mendeley_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[ImportSample] = []
    for image_id, anns in ann_by_image.items():
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

        symbols = []
        x_min = 10**9
        y_min = 10**9
        x_max = 0.0
        y_max = 0.0
        for a in anns:
            bbox = a.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            sym = category_to_symbol.get(int(a.get("category_id", -1)), "")
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
        if max_images is not None and len(samples) >= max_images:
            break

    return samples


def build_hf_samples(max_images: int | None = None) -> list[ImportSample]:
    from datasets import load_dataset

    out_dir = STAGING_DIR / "hf_7seg_ocr_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_DATASET_ID, split="train")
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
        if max_images is not None and len(samples) >= max_images:
            break

    return samples


def parse_roboflow_yolo_export(dataset_root: Path, max_images: int | None = None) -> list[ImportSample]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"Missing data.yaml in {dataset_root}")
    with open(data_yaml, "r", encoding="utf-8") as fh:
        ycfg = yaml.safe_load(fh)

    names = ycfg.get("names", [])
    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    else:
        id_to_name = {i: str(v) for i, v in enumerate(names)}

    samples: list[ImportSample] = []
    splits = ["train", "valid", "test"]
    for split in splits:
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
            w, h = read_image_size(image_path)
            detections = []
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                x1 = max(0.0, xc - bw / 2.0)
                y1 = max(0.0, yc - bh / 2.0)
                x2 = min(float(w), xc + bw / 2.0)
                y2 = min(float(h), yc + bh / 2.0)
                sym = guess_symbol(id_to_name.get(cls_id, str(cls_id)))
                detections.append((xc, sym, x1, y1, x2, y2))
            if not detections:
                continue
            detections.sort(key=lambda t: t[0])
            gt = sanitize_label("".join(d[1] for d in detections))
            if not gt:
                continue
            x_min = int(max(0.0, min(d[2] for d in detections)))
            y_min = int(max(0.0, min(d[3] for d in detections)))
            x_max = int(min(float(w), max(d[4] for d in detections)))
            y_max = int(min(float(h), max(d[5] for d in detections)))
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
) -> list[ImportSample]:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required to download Roboflow datasets from this environment.")

    from roboflow import Roboflow

    out_dir = EXTERNAL_DIR / local_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    versions = project.versions()
    if not versions:
        raise RuntimeError(f"No versions found for {workspace}/{project_slug}")
    latest_version = max(int(v.version) for v in versions)
    dataset = project.version(latest_version).download("yolov8", location=str(out_dir))
    dataset_root = Path(dataset.location)
    return parse_roboflow_yolo_export(dataset_root, max_images=max_images)


def batched(items: list[ImportSample], batch_size: int) -> Iterable[list[ImportSample]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def ensure_unique_dataset_name(base_name: str) -> str:
    return f"{base_name}_{now_tag()}_{uuid.uuid4().hex[:6]}"


def import_into_app_and_run(
    app,
    dataset_name: str,
    samples: list[ImportSample],
    pipeline_slugs: list[str],
    upload_batch_size: int = 20,
    poll_seconds: float = 2.0,
) -> dict:
    if not samples:
        raise RuntimeError("No samples to import.")

    client = app.test_client()
    dataset_name = ensure_unique_dataset_name(dataset_name)

    resp = client.post(
        "/upload/dataset",
        data={
            "name": dataset_name,
            "description": f"Auto-imported external dataset at {datetime.now().isoformat(timespec='seconds')}",
            "lighting_tag": "mixed",
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed creating dataset {dataset_name}: {resp.status_code} {resp.data!r}")
    dataset_id = resp.get_json()["id"]

    uploaded_total = 0
    upload_errors = 0
    for batch in batched(samples, upload_batch_size):
        open_files = []
        try:
            files_payload = []
            for s in batch:
                fh = open(s.image_path, "rb")
                open_files.append(fh)
                files_payload.append((fh, s.filename))
            resp = client.post(
                "/upload/images",
                data={"dataset_id": str(dataset_id), "files": files_payload},
                content_type="multipart/form-data",
            )
        finally:
            for fh in open_files:
                fh.close()
        if resp.status_code != 200:
            raise RuntimeError(f"Image upload failed for dataset {dataset_name}: {resp.status_code} {resp.data!r}")
        payload = resp.get_json()
        for row in payload.get("uploaded", []):
            if "error" in row:
                upload_errors += 1
            else:
                uploaded_total += 1

    labels = [
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
    label_bytes = io.BytesIO(json.dumps(labels, ensure_ascii=True).encode("utf-8"))
    label_bytes.name = f"{dataset_name}_labels.json"
    resp = client.post(
        "/label/import",
        data={"dataset_id": str(dataset_id), "file": (label_bytes, label_bytes.name)},
        content_type="multipart/form-data",
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Label import failed for dataset {dataset_name}: {resp.status_code} {resp.data!r}")
    imported_count = resp.get_json().get("imported", 0)

    resp = client.post(
        "/benchmark/start",
        json={
            "dataset_id": dataset_id,
            "pipeline_slugs": pipeline_slugs,
            "name": f"{dataset_name}_{'_'.join(pipeline_slugs)}",
            "pipeline_configs": {},
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Benchmark start failed for dataset {dataset_name}: {resp.status_code} {resp.data!r}")
    run_id = resp.get_json()["run_id"]

    status_payload = {}
    while True:
        st = client.get(f"/benchmark/status/{run_id}")
        if st.status_code != 200:
            raise RuntimeError(f"Failed polling benchmark status for run {run_id}")
        status_payload = st.get_json()
        status = status_payload.get("status")
        processed = status_payload.get("processed")
        total = status_payload.get("total")
        print(f"[run {run_id}] status={status} processed={processed}/{total}")
        if status in {"completed", "failed"}:
            break
        time.sleep(poll_seconds)

    with app.app_context():
        run = BenchmarkRun.query.get(run_id)
        summary = run.summary if run else {}

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "uploaded_images": uploaded_total,
        "upload_errors": upload_errors,
        "labels_imported": imported_count,
        "run_id": run_id,
        "run_status": status_payload.get("status"),
        "run_processed": status_payload.get("processed"),
        "run_total": status_payload.get("total"),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images-per-dataset", type=int, default=None)
    parser.add_argument("--pipelines", nargs="+", default=["p01_global_threshold"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["roboflow1", "roboflow2", "hf", "mendeley"],
        help="Subset: roboflow1 roboflow2 hf mendeley",
    )
    args = parser.parse_args()

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    app = create_app("dev")
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipelines": args.pipelines,
        "max_images_per_dataset": args.max_images_per_dataset,
        "results": {},
    }

    jobs = []
    if "roboflow1" in args.datasets:
        jobs.append(("roboflow1", "rf_seven_segment_digits", lambda: build_roboflow_samples("charlie-srmko", "seven-segment-digits-uptcy", "roboflow_seven_segment_digits", args.max_images_per_dataset)))
    if "roboflow2" in args.datasets:
        jobs.append(("roboflow2", "rf_seven_segment_display_ocr", lambda: build_roboflow_samples("fyp-zodww", "seven-segment-display-ocr-lguqw", "roboflow_seven_segment_display_ocr", args.max_images_per_dataset)))
    if "hf" in args.datasets:
        jobs.append(("hf", "hf_7seg_ocr", lambda: build_hf_samples(args.max_images_per_dataset)))
    if "mendeley" in args.datasets:
        jobs.append(("mendeley", "mendeley_fnn44p4mj8", lambda: build_mendeley_samples(args.max_images_per_dataset)))

    for key, app_name, loader in jobs:
        print(f"\n=== Processing {key} ===")
        try:
            samples = loader()
            print(f"{key}: prepared {len(samples)} labeled samples")
            result = import_into_app_and_run(
                app=app,
                dataset_name=app_name,
                samples=samples,
                pipeline_slugs=args.pipelines,
            )
            report["results"][key] = {"ok": True, **result}
            print(f"{key}: completed run {result['run_id']} status={result['run_status']}")
        except Exception as exc:
            report["results"][key] = {"ok": False, "error": str(exc)}
            print(f"{key}: FAILED - {exc}")

    report_path = EXTERNAL_DIR / f"install_and_run_report_{now_tag()}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
