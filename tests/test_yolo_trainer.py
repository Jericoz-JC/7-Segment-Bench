"""Unit tests for YOLO trainer helpers."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app
from models import db


@pytest.fixture
def app():
    app = create_app('dev')
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


def _make_labeled_dataset():
    from models.dataset import Dataset
    from models.image import Image
    from models.label import Label

    ds = Dataset(name='yolo_trainer_ds')
    db.session.add(ds)
    db.session.commit()

    for i in range(6):
        img = Image(
            dataset_id=ds.id,
            filename=f'img_{i}.png',
            filepath=f'img_{i}.png',
            width=320 + (i * 10),
            height=120,
        )
        db.session.add(img)
        db.session.commit()
        lbl = Label(
            image_id=img.id,
            roi_x=10,
            roi_y=20,
            roi_width=240,
            roi_height=80,
            ground_truth='1234',
            num_digits=4,
        )
        db.session.add(lbl)
        db.session.commit()

    return ds.id


def test_export_yolo_dataset_uses_split_builder(monkeypatch):
    import services.yolo_trainer as yolo_trainer

    captured = {}

    def fake_build(dataset_id, output_dir, val_ratio=0.2, seed=42):
        captured['dataset_id'] = dataset_id
        captured['output_dir'] = output_dir
        captured['val_ratio'] = val_ratio
        captured['seed'] = seed
        return {'yaml_path': 'C:/tmp/split/data.yaml'}

    monkeypatch.setattr(yolo_trainer, 'build_split_yolo_dataset', fake_build)
    out = yolo_trainer.export_yolo_dataset(12, 'C:/tmp/out', val_ratio=0.3, seed=7)
    assert out == 'C:/tmp/split/data.yaml'
    assert captured['dataset_id'] == 12
    assert captured['output_dir'] == 'C:/tmp/out'
    assert captured['val_ratio'] == 0.3
    assert captured['seed'] == 7


def test_recommend_training_params_returns_expected_shape(app):
    import services.yolo_trainer as yolo_trainer

    with app.app_context():
        dataset_id = _make_labeled_dataset()
        payload = yolo_trainer.recommend_training_params(
            dataset_id=dataset_id,
            base_model='yolov8s.pt',
            device='cpu',
        )

    assert payload['dataset_id'] == dataset_id
    assert payload['stats']['labeled_count'] == 6
    assert payload['recommended']['base_model'] == 'yolov8s.pt'
    assert 1 <= payload['recommended']['epochs'] <= 500
    assert 128 <= payload['recommended']['imgsz'] <= 2048
    assert 1 <= payload['recommended']['batch'] <= 128
    assert payload['recommended']['seed'] == 42


def test_recommend_training_params_rejects_invalid_model(app):
    import services.yolo_trainer as yolo_trainer

    with app.app_context():
        dataset_id = _make_labeled_dataset()
        with pytest.raises(ValueError):
            yolo_trainer.recommend_training_params(
                dataset_id=dataset_id,
                base_model='bad.pt',
                device='cpu',
            )
