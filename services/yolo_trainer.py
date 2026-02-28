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


def export_yolo_dataset(dataset_id: int, output_dir: str) -> str:
    """Backward-compatible flat YOLO export (train=val=images)."""
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        raise ValueError(f'Dataset {dataset_id} not found')

    out_path = Path(output_dir)
    images_dir = out_path / 'images'
    labels_dir = out_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img, label in _get_labeled_pairs(dataset_id):
        stem = Path(img.filepath).stem
        suffix = Path(img.filepath).suffix or '.png'
        target_name = f'{img.id}_{stem}{suffix}'
        target_img = images_dir / target_name
        target_lbl = labels_dir / f'{img.id}_{stem}.txt'
        shutil.copy2(get_image_path(img), target_img)
        target_lbl.write_text('\n'.join(_digit_label_lines(img, label)), encoding='utf-8')

    data_yaml = {
        'path': str(out_path.resolve()),
        'train': 'images',
        'val': 'images',
        'names': {i: str(i) for i in range(10)},
    }
    yaml_path = out_path / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(data_yaml), encoding='utf-8')
    return str(yaml_path)


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


def train_yolo_model(
    data_yaml_path: str,
    project_dir: str,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None = None,
) -> dict:
    """Train a YOLOv8 model and return artifact metadata."""
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
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
