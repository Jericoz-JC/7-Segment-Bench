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
