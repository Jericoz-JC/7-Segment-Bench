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
