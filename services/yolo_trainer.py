"""YOLO dataset export and training utilities."""

from __future__ import annotations

import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from models import db
from models.benchmark import TrainedModel
from models.dataset import Dataset
from models.image import Image
from models.label import Label
from services.image_service import get_image_path

YOLO_BASE_MODELS = ('yolov8n.pt', 'yolov8s.pt')


def _digit_label_lines(img: Image, label: Label) -> list[str]:
    gt = label.ground_truth or ''
    num_chars = len(gt)
    if num_chars == 0:
        return []

    roi_width = label.roi_width or img.width
    roi_height = label.roi_height or img.height
    if roi_width <= 0 or roi_height <= 0 or img.width <= 0 or img.height <= 0:
        return []

    digit_width = roi_width / num_chars
    lines = []
    for i, ch in enumerate(gt):
        if not ch.isdigit():
            continue
        cls_id = int(ch)
        cx = (label.roi_x + digit_width * (i + 0.5)) / img.width
        cy = (label.roi_y + roi_height / 2) / img.height
        w = digit_width / img.width
        h = roi_height / img.height
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return lines


def _copy_sample_to_split(img: Image, label: Label, out_dir: Path, split: str):
    split_img_dir = out_dir / split / 'images'
    split_lbl_dir = out_dir / split / 'labels'
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(img.filepath).stem
    suffix = Path(img.filepath).suffix or '.png'
    target_name = f'{img.id}_{stem}{suffix}'
    target_img = split_img_dir / target_name
    target_lbl = split_lbl_dir / f'{img.id}_{stem}.txt'

    src = get_image_path(img)
    shutil.copy2(src, target_img)
    lines = _digit_label_lines(img, label)
    target_lbl.write_text('\n'.join(lines), encoding='utf-8')


def _get_labeled_pairs(dataset_id: int) -> list[tuple[Image, Label]]:
    return (
        db.session.query(Image, Label)
        .join(Label)
        .filter(Image.dataset_id == dataset_id)
        .all()
    )


def export_yolo_dataset(
    dataset_id: int,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> str:
    """Export a dataset into train/val YOLO format and return data.yaml path."""
    split_info = build_split_yolo_dataset(
        dataset_id=dataset_id,
        output_dir=output_dir,
        val_ratio=val_ratio,
        seed=seed,
    )
    return split_info['yaml_path']


def build_split_yolo_dataset(
    dataset_id: int,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Export a single dataset into train/val YOLO format."""
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        raise ValueError(f'Dataset {dataset_id} not found')

    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError('val_ratio must be between 0 and 1 (exclusive)')

    labeled = _get_labeled_pairs(dataset_id)
    if len(labeled) < 2:
        raise ValueError('Need at least 2 labeled images for train/val split')

    rng = random.Random(seed)
    rng.shuffle(labeled)

    val_count = max(1, int(round(len(labeled) * val_ratio)))
    if val_count >= len(labeled):
        val_count = len(labeled) - 1

    val_items = labeled[:val_count]
    train_items = labeled[val_count:]

    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for img, label in train_items:
        _copy_sample_to_split(img, label, out_path, 'train')
    for img, label in val_items:
        _copy_sample_to_split(img, label, out_path, 'val')

    data_yaml = {
        'path': str(out_path.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: str(i) for i in range(10)},
    }
    yaml_path = out_path / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(data_yaml), encoding='utf-8')

    return {
        'dataset_id': dataset_id,
        'dataset_name': dataset.name,
        'yaml_path': str(yaml_path),
        'train_count': len(train_items),
        'val_count': len(val_items),
        'total_count': len(labeled),
    }


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return int(ordered[mid])
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def _normalize_device_hint(device: str) -> str:
    value = (device or '').strip().lower()
    if not value or value == 'auto':
        return 'auto'
    if value.isdigit() or value.startswith('cuda') or value.startswith('gpu'):
        return 'cuda'
    if value.startswith('mps'):
        return 'mps'
    if 'cpu' in value:
        return 'cpu'
    return 'auto'


def _detect_gpu_vram_gb() -> int:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(0)
        return int(round(props.total_memory / (1024 ** 3)))
    except Exception:
        return 0


def recommend_training_params(
    dataset_id: int,
    base_model: str = 'yolov8s.pt',
    device: str = '',
) -> dict:
    """Return explainable, deterministic YOLO training parameter suggestions."""
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        raise ValueError(f'Dataset {dataset_id} not found')
    if base_model not in YOLO_BASE_MODELS:
        raise ValueError(f'base_model must be one of: {", ".join(YOLO_BASE_MODELS)}')

    rows = (
        db.session.query(Image, Label)
        .join(Label)
        .filter(Image.dataset_id == dataset_id)
        .all()
    )
    if not rows:
        raise ValueError('Need labeled images before generating recommendations')

    image_long_sides: list[int] = []
    roi_heights: list[int] = []
    digit_counts: list[int] = []

    for img, label in rows:
        width = int(img.width or 0)
        height = int(img.height or 0)
        if width > 0 and height > 0:
            image_long_sides.append(max(width, height))

        roi_h = int(label.roi_height or 0)
        if roi_h <= 0:
            roi_h = height
        if roi_h > 0:
            roi_heights.append(roi_h)

        digits = sum(ch.isdigit() for ch in (label.ground_truth or ''))
        digit_counts.append(digits if digits > 0 else int(label.num_digits or 0) or 1)

    labeled_count = len(rows)
    median_long_side = _median(image_long_sides)
    median_roi_height = _median(roi_heights)
    median_digits = max(1, _median(digit_counts))

    if median_long_side <= 0:
        median_long_side = 640
    if median_roi_height <= 0:
        median_roi_height = max(32, median_long_side // 3)

    suggested_target = max(int(median_long_side * 0.75), median_roi_height * max(2, median_digits))
    if suggested_target <= 256:
        imgsz = 320
    elif suggested_target <= 448:
        imgsz = 512
    else:
        imgsz = 640

    if labeled_count >= 5000:
        epochs = 36
    elif labeled_count >= 2500:
        epochs = 44
    elif labeled_count >= 1000:
        epochs = 58
    elif labeled_count >= 300:
        epochs = 80
    else:
        epochs = 110
    if base_model == 'yolov8s.pt':
        epochs = max(20, epochs - 8)

    normalized_device = _normalize_device_hint(device)
    gpu_like = normalized_device in ('cuda', 'mps')
    vram_gb = _detect_gpu_vram_gb() if normalized_device == 'cuda' else 0

    base_batch = 10 if base_model == 'yolov8n.pt' else 6
    if not gpu_like:
        base_batch = max(2, base_batch - 3)
    if vram_gb >= 12:
        base_batch += 4
    elif vram_gb >= 8:
        base_batch += 2
    if imgsz >= 640:
        base_batch -= 2
    elif imgsz <= 320:
        base_batch += 2
    batch = max(1, min(128, int(base_batch)))

    notes = [
        'Auto-suggest uses dataset-size and ROI-scale heuristics; keep manual overrides for final tuning.',
        f'Chosen base model: {base_model}.',
        f'Device hint resolved to: {normalized_device}.',
    ]
    if normalized_device == 'cuda':
        if vram_gb > 0:
            notes.append(f'Detected CUDA GPU memory: ~{vram_gb} GB.')
        else:
            notes.append('CUDA selected, but GPU memory could not be detected; using conservative batch.')

    return {
        'dataset_id': dataset_id,
        'dataset_name': dataset.name,
        'stats': {
            'labeled_count': labeled_count,
            'median_image_long_side': median_long_side,
            'median_roi_height': median_roi_height,
            'median_num_digits': median_digits,
        },
        'recommended': {
            'epochs': int(epochs),
            'imgsz': int(imgsz),
            'batch': int(batch),
            'seed': 42,
            'base_model': base_model,
        },
        'notes': notes,
    }


def train_yolo_model(
    data_yaml_path: str,
    project_dir: str,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None = None,
    base_model: str = 'yolov8s.pt',
) -> dict:
    """Train a YOLOv8 model and return artifact metadata."""
    from ultralytics import YOLO

    if base_model not in YOLO_BASE_MODELS:
        raise ValueError(f'base_model must be one of: {", ".join(YOLO_BASE_MODELS)}')

    model = YOLO(base_model)
    kwargs = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'project': project_dir,
        'name': run_name,
        'exist_ok': False,
        'verbose': False,
    }
    if device:
        kwargs['device'] = device

    results = model.train(**kwargs)
    save_dir = Path(results.save_dir)
    best_path = save_dir / 'weights' / 'best.pt'
    if not best_path.exists():
        raise RuntimeError(f'Expected trained model at {best_path}, but it was not produced')

    metrics = {}
    if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
        metrics = results.results_dict

    return {
        'save_dir': str(save_dir),
        'best_model_path': str(best_path),
        'metrics': metrics,
    }


def list_trained_models(pipeline_slug: str = 'p06_yolo_nano') -> list[TrainedModel]:
    return (
        TrainedModel.query
        .filter_by(pipeline_slug=pipeline_slug)
        .order_by(TrainedModel.created_at.desc())
        .all()
    )


def register_trained_model(
    dataset_id: int,
    model_path: str,
    run_name: str,
    metrics: dict | None = None,
    pipeline_slug: str = 'p06_yolo_nano',
    activate: bool = True,
) -> TrainedModel:
    if activate:
        TrainedModel.query.filter_by(pipeline_slug=pipeline_slug).update({'is_active': False})

    item = TrainedModel(
        name=run_name,
        pipeline_slug=pipeline_slug,
        dataset_id=dataset_id,
        model_path=model_path,
        is_active=activate,
    )
    item.metrics = metrics or {}
    db.session.add(item)
    db.session.commit()
    return item


def activate_trained_model(model_id: int, pipeline_slug: str = 'p06_yolo_nano') -> TrainedModel:
    model = TrainedModel.query.get_or_404(model_id)
    TrainedModel.query.filter_by(pipeline_slug=pipeline_slug).update({'is_active': False})
    model.is_active = True
    db.session.commit()
    return model


def get_active_trained_model(pipeline_slug: str = 'p06_yolo_nano') -> TrainedModel | None:
    return (
        TrainedModel.query
        .filter_by(pipeline_slug=pipeline_slug, is_active=True)
        .order_by(TrainedModel.created_at.desc())
        .first()
    )


def default_yolo_training_root() -> str:
    base_dir = Path(__file__).resolve().parent.parent
    out = base_dir / 'data' / 'training' / 'p06_yolo_nano'
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def make_training_run_name(dataset_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    safe_dataset = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in dataset_name)
    return f'{safe_dataset}_{ts}'
