"""Integration tests for Flask routes."""

import pytest
import sys
import os
import io

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


@pytest.fixture
def client(app):
    return app.test_client()


class TestDashboard:
    def test_index(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert b'Dashboard' in resp.data


class TestUpload:
    def test_upload_page(self, client):
        resp = client.get('/upload/')
        assert resp.status_code == 200
        assert b'Upload' in resp.data
        assert b'Curated External Datasets' in resp.data
        assert b'Quick Add' in resp.data

    def test_external_dataset_catalog_endpoint(self, client):
        resp = client.get('/upload/external-datasets')
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        keys = {item['key'] for item in data}
        assert 'hf_7seg_ocr' in keys
        assert 'mendeley_fnn44p4mj8' in keys
        assert 'roboflow_seven_segment_digits' in keys
        first = data[0]
        assert 'install_allowed' in first
        assert 'install_reason' in first
        assert 'requires_modules' in first

    def test_create_dataset(self, client):
        resp = client.post('/upload/dataset', data={
            'name': 'test_dataset',
            'description': 'Test',
            'lighting_tag': 'bright',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['name'] == 'test_dataset'
        assert 'id' in data

    def test_create_duplicate_dataset(self, client):
        client.post('/upload/dataset', data={'name': 'dup'})
        resp = client.post('/upload/dataset', data={'name': 'dup'})
        assert resp.status_code == 400

    def test_external_dataset_install_rejects_unknown_key(self, client):
        resp = client.post('/upload/external-datasets/install', json={
            'dataset_key': 'does-not-exist',
            'mode': 'quick',
        })
        assert resp.status_code == 400
        assert 'Unknown dataset key' in resp.get_json()['error']

    def test_external_dataset_install_rejects_invalid_mode(self, client):
        resp = client.post('/upload/external-datasets/install', json={
            'dataset_key': 'hf_7seg_ocr',
            'mode': 'invalid',
        })
        assert resp.status_code == 400
        assert "mode must be either 'quick' or 'full'" in resp.get_json()['error']

    def test_external_dataset_install_requires_roboflow_key(self, client, monkeypatch):
        monkeypatch.delenv('ROBOFLOW_API_KEY', raising=False)
        resp = client.post('/upload/external-datasets/install', json={
            'dataset_key': 'roboflow_seven_segment_digits',
            'mode': 'quick',
        })
        assert resp.status_code == 400
        assert 'ROBOFLOW_API_KEY' in resp.get_json()['error']

    def test_external_dataset_install_requires_hf_package(self, client, monkeypatch):
        import services.external_dataset_catalog as catalog

        real_find_spec = catalog.importlib.util.find_spec

        def fake_find_spec(name):
            if name == 'datasets':
                return None
            return real_find_spec(name)

        monkeypatch.setattr(catalog.importlib.util, 'find_spec', fake_find_spec)
        resp = client.post('/upload/external-datasets/install', json={
            'dataset_key': 'hf_7seg_ocr',
            'mode': 'quick',
        })
        assert resp.status_code == 400
        assert "Python package 'datasets' is required" in resp.get_json()['error']

    def test_external_dataset_install_starts_job(self, client, monkeypatch):
        import routes.upload as upload_routes

        captured = {}

        def fake_start_install(app_obj, dataset_key, mode):
            captured['dataset_key'] = dataset_key
            captured['mode'] = mode
            return 'job_test_123'

        monkeypatch.setattr(upload_routes, 'validate_install_request', lambda dataset_key, mode: None)
        monkeypatch.setattr(upload_routes._install_manager, 'start_install', fake_start_install)

        resp = client.post('/upload/external-datasets/install', json={
            'dataset_key': 'hf_7seg_ocr',
            'mode': 'quick',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['job_id'] == 'job_test_123'
        assert captured['dataset_key'] == 'hf_7seg_ocr'
        assert captured['mode'] == 'quick'

    def test_external_dataset_status_not_found(self, client):
        resp = client.get('/upload/external-datasets/status/not-a-job')
        assert resp.status_code == 404

    def test_external_dataset_status_success(self, client, monkeypatch):
        import routes.upload as upload_routes

        payload = {
            'job_id': 'job_test_456',
            'dataset_key': 'hf_7seg_ocr',
            'mode': 'quick',
            'status': 'running',
            'phase': 'import',
            'processed': 10,
            'total': 100,
            'progress_pct': 10.0,
            'message': 'Importing 10/100 images',
            'result': None,
            'error': '',
        }

        monkeypatch.setattr(upload_routes._install_manager, 'get_status', lambda job_id: payload)
        resp = client.get('/upload/external-datasets/status/job_test_456')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['job_id'] == 'job_test_456'
        assert data['status'] == 'running'


class TestLabel:
    def test_label_page(self, client):
        resp = client.get('/label/')
        assert resp.status_code == 200

    def test_save_label(self, client):
        # Create dataset and image first
        from models.dataset import Dataset
        from models.image import Image
        with client.application.app_context():
            ds = Dataset(name='test_ds')
            db.session.add(ds)
            db.session.commit()
            img = Image(dataset_id=ds.id, filename='test.png',
                        filepath='test.png', width=640, height=480)
            db.session.add(img)
            db.session.commit()
            img_id = img.id

        resp = client.post('/label/save', json={
            'image_id': img_id,
            'ground_truth': '1234',
            'roi_x': 0, 'roi_y': 0, 'roi_width': 640, 'roi_height': 480,
            'display_type': 'led',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ground_truth'] == '1234'


class TestBenchmark:
    def test_benchmark_page(self, client):
        resp = client.get('/benchmark/')
        assert resp.status_code == 200


class TestResults:
    def test_results_page(self, client):
        resp = client.get('/results/')
        assert resp.status_code == 200


class TestSingleTest:
    def test_test_page(self, client):
        resp = client.get('/test/')
        assert resp.status_code == 200

    def test_run_batch_rejects_invalid_sample_size(self, client):
        resp = client.post('/test/run-batch', json={
            'dataset_id': 1,
            'pipeline_slugs': ['p01_global_threshold'],
            'sample_size': 5,
        })
        assert resp.status_code == 400
        assert 'sample_size must be between 10 and 50' in resp.get_json()['error']

    def test_run_batch_starts_run(self, client, monkeypatch):
        from models.dataset import Dataset
        from models.image import Image
        from models.label import Label
        import routes.single_test as single_test_routes

        with client.application.app_context():
            ds = Dataset(name='batch_ds')
            db.session.add(ds)
            db.session.commit()

            for i in range(3):
                img = Image(
                    dataset_id=ds.id,
                    filename=f'i{i}.png',
                    filepath=f'i{i}.png',
                    width=200,
                    height=80,
                )
                db.session.add(img)
                db.session.commit()
                lbl = Label(
                    image_id=img.id,
                    roi_x=0,
                    roi_y=0,
                    roi_width=200,
                    roi_height=80,
                    ground_truth='1234',
                    num_digits=4,
                )
                db.session.add(lbl)
                db.session.commit()

            ds_id = ds.id

        captured = {}

        def fake_launch(run_id):
            captured['run_id'] = run_id
            return None

        monkeypatch.setattr(single_test_routes, 'launch_runner', fake_launch)

        resp = client.post('/test/run-batch', json={
            'dataset_id': ds_id,
            'pipeline_slugs': ['p01_global_threshold'],
            'sample_size': 10,
            'name': 'quick_batch_test',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'run_id' in data
        assert data['selected_count'] == 3
        assert len(data['selected_image_ids']) == 3
        assert captured['run_id'] == data['run_id']

    def test_run_single_passes_pipeline_configs(self, client, monkeypatch):
        import json
        import cv2
        import numpy as np
        import routes.single_test as single_test_routes
        from pipelines.base import PipelineResult

        captured = {}

        class FakePipe:
            def load(self):
                pass

            def unload(self):
                pass

            def predict_timed(self, image, roi):
                return PipelineResult(predicted='1234', confidence=0.9, latency_ms=1.2)

        def fake_get_pipeline(slug, config=None):
            captured['slug'] = slug
            captured['config'] = config
            return FakePipe()

        monkeypatch.setattr(single_test_routes.pipelines, 'get_pipeline', fake_get_pipeline)

        image = np.zeros((40, 120, 3), dtype=np.uint8)
        ok, buf = cv2.imencode('.png', image)
        assert ok is True

        payload = {
            'image': (io.BytesIO(buf.tobytes()), 'single.png'),
            'pipelines': 'p07_llm_vision',
            'pipeline_configs': json.dumps({
                'p07_llm_vision': {
                    'provider': 'openrouter',
                    'model': 'openrouter/auto',
                }
            }),
        }
        resp = client.post('/test/run', data=payload, content_type='multipart/form-data')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'p07_llm_vision' in data
        assert captured['slug'] == 'p07_llm_vision'
        assert captured['config']['provider'] == 'openrouter'
        assert captured['config']['model'] == 'openrouter/auto'


class TestAPI:
    def test_list_keys(self, client):
        resp = client.get('/api/keys')
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_add_key(self, client):
        resp = client.post('/api/keys', json={
            'provider': 'anthropic',
            'key_value': 'sk-test-12345678',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['provider'] == 'anthropic'

    def test_invalid_provider(self, client):
        resp = client.post('/api/keys', json={
            'provider': 'invalid',
            'key_value': 'test',
        })
        assert resp.status_code == 400

    def test_add_openrouter_key(self, client):
        resp = client.post('/api/keys', json={
            'provider': 'openrouter',
            'key_value': 'or-test-123456',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['provider'] == 'openrouter'

    def test_add_ollama_without_key(self, client):
        resp = client.post('/api/keys', json={
            'provider': 'ollama',
            'key_value': '',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['provider'] == 'ollama'

    def test_p05_preflight_endpoint(self, client):
        resp = client.get('/api/pipelines/p05/preflight')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'ok' in data
        assert 'version' in data
        assert 'error' in data


class TestExport:
    def test_export_labels_empty(self, client):
        from models.dataset import Dataset
        with client.application.app_context():
            ds = Dataset(name='export_test')
            db.session.add(ds)
            db.session.commit()
            ds_id = ds.id
        resp = client.get(f'/export/labels/{ds_id}')
        assert resp.status_code == 200
        assert resp.get_json() == []


class TestYoloTrainRoutes:
    def test_train_page(self, client):
        resp = client.get('/train/yolo/')
        assert resp.status_code == 200
        assert b'YOLO Training Lab' in resp.data

    def test_train_status_not_found(self, client):
        resp = client.get('/train/yolo/status/not-a-job')
        assert resp.status_code == 404

    def test_train_start_starts_job(self, client, monkeypatch):
        import routes.train_yolo as train_routes
        from models.dataset import Dataset
        from models.image import Image
        from models.label import Label

        with client.application.app_context():
            ds = Dataset(name='train_ds')
            db.session.add(ds)
            db.session.commit()

            for i in range(2):
                img = Image(
                    dataset_id=ds.id,
                    filename=f'train{i}.png',
                    filepath=f'train{i}.png',
                    width=240,
                    height=90,
                )
                db.session.add(img)
                db.session.commit()
                lbl = Label(
                    image_id=img.id,
                    roi_x=0,
                    roi_y=0,
                    roi_width=240,
                    roi_height=90,
                    ground_truth='1234',
                    num_digits=4,
                )
                db.session.add(lbl)
                db.session.commit()
            ds_id = ds.id

        captured = {}

        def fake_start_training(app_obj, dataset_id, params):
            captured['dataset_id'] = dataset_id
            captured['params'] = params
            return 'job_train_1'

        monkeypatch.setattr(train_routes._train_manager, 'start_training', fake_start_training)

        resp = client.post('/train/yolo/start', json={
            'dataset_id': ds_id,
            'base_model': 'yolov8s.pt',
            'epochs': 5,
            'imgsz': 320,
            'batch': 2,
            'val_ratio': 0.2,
            'seed': 42,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['job_id'] == 'job_train_1'
        assert captured['dataset_id'] == ds_id
        assert captured['params']['base_model'] == 'yolov8s.pt'

    def test_train_start_rejects_invalid_base_model(self, client):
        from models.dataset import Dataset
        from models.image import Image
        from models.label import Label

        with client.application.app_context():
            ds = Dataset(name='train_invalid_base')
            db.session.add(ds)
            db.session.commit()

            for i in range(2):
                img = Image(
                    dataset_id=ds.id,
                    filename=f'invalid{i}.png',
                    filepath=f'invalid{i}.png',
                    width=240,
                    height=90,
                )
                db.session.add(img)
                db.session.commit()
                lbl = Label(
                    image_id=img.id,
                    roi_x=0,
                    roi_y=0,
                    roi_width=240,
                    roi_height=90,
                    ground_truth='1234',
                    num_digits=4,
                )
                db.session.add(lbl)
                db.session.commit()
            ds_id = ds.id

        resp = client.post('/train/yolo/start', json={
            'dataset_id': ds_id,
            'base_model': 'bad-model.pt',
            'epochs': 5,
            'imgsz': 320,
            'batch': 2,
            'val_ratio': 0.2,
            'seed': 42,
        })
        assert resp.status_code == 400
        assert 'base_model must be one of' in resp.get_json()['error']

    def test_train_recommend_success(self, client, monkeypatch):
        import routes.train_yolo as train_routes

        monkeypatch.setattr(
            train_routes,
            'recommend_training_params',
            lambda dataset_id, base_model, device: {
                'dataset_id': dataset_id,
                'dataset_name': 'demo',
                'stats': {'labeled_count': 10},
                'recommended': {
                    'epochs': 40,
                    'imgsz': 640,
                    'batch': 4,
                    'seed': 42,
                    'base_model': base_model,
                },
                'notes': ['ok'],
            },
        )

        resp = client.post('/train/yolo/recommend', json={
            'dataset_id': 1,
            'base_model': 'yolov8s.pt',
            'device': 'cpu',
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['recommended']['epochs'] == 40
        assert data['recommended']['base_model'] == 'yolov8s.pt'

    def test_train_recommend_rejects_invalid_base_model(self, client):
        resp = client.post('/train/yolo/recommend', json={
            'dataset_id': 1,
            'base_model': 'bad-model.pt',
        })
        assert resp.status_code == 400
        assert 'base_model must be one of' in resp.get_json()['error']

    def test_activate_model(self, client):
        from models.dataset import Dataset
        from models.benchmark import TrainedModel

        with client.application.app_context():
            ds = Dataset(name='activate_ds')
            db.session.add(ds)
            db.session.commit()
            model = TrainedModel(
                name='m1',
                pipeline_slug='p06_yolo_nano',
                dataset_id=ds.id,
                model_path='C:/tmp/best.pt',
                is_active=False,
            )
            db.session.add(model)
            db.session.commit()
            model_id = model.id

        resp = client.post(f'/train/yolo/models/{model_id}/activate')
        assert resp.status_code == 200
        assert resp.get_json()['ok'] is True
