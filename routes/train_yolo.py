from flask import Blueprint, current_app, render_template, request, jsonify

import pipelines
from models.dataset import Dataset
from services.yolo_trainer import (
    YOLO_BASE_MODELS,
    activate_trained_model,
    list_trained_models,
    recommend_training_params,
)
from services.yolo_training_jobs import YoloTrainingJobManager

bp = Blueprint('train_yolo', __name__)
_train_manager = YoloTrainingJobManager()


@bp.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    labeled_datasets = [d for d in datasets if d.labeled_count > 0]
    models = list_trained_models()
    return render_template(
        'train_yolo.html',
        datasets=labeled_datasets,
        models=models,
        pipelines=pipelines.list_pipelines(),
        yolo_base_models=YOLO_BASE_MODELS,
    )


@bp.route('/start', methods=['POST'])
def start():
    data = request.get_json() or {}

    dataset_id = data.get('dataset_id')
    if dataset_id is None:
        return jsonify({'error': 'dataset_id is required'}), 400
    try:
        dataset_id = int(dataset_id)
    except Exception:
        return jsonify({'error': 'dataset_id must be an integer'}), 400

    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.labeled_count < 2:
        return jsonify({'error': 'Need at least 2 labeled images to train'}), 400

    try:
        epochs = int(data.get('epochs', 30))
        imgsz = int(data.get('imgsz', 640))
        batch = int(data.get('batch', 8))
        seed = int(data.get('seed', 42))
        val_ratio = float(data.get('val_ratio', 0.2))
    except Exception:
        return jsonify({'error': 'Invalid numeric training config'}), 400

    if epochs < 1 or epochs > 500:
        return jsonify({'error': 'epochs must be between 1 and 500'}), 400
    if imgsz < 128 or imgsz > 2048:
        return jsonify({'error': 'imgsz must be between 128 and 2048'}), 400
    if batch < 1 or batch > 128:
        return jsonify({'error': 'batch must be between 1 and 128'}), 400
    if val_ratio <= 0 or val_ratio >= 1:
        return jsonify({'error': 'val_ratio must be between 0 and 1'}), 400

    run_name = str(data.get('run_name', '')).strip()
    device = str(data.get('device', '')).strip()
    base_model = str(data.get('base_model', 'yolov8s.pt')).strip()
    activate = bool(data.get('activate', True))

    if base_model not in YOLO_BASE_MODELS:
        return jsonify({'error': f'base_model must be one of: {", ".join(YOLO_BASE_MODELS)}'}), 400

    params = {
        'run_name': run_name,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'seed': seed,
        'val_ratio': val_ratio,
        'device': device,
        'base_model': base_model,
        'activate': activate,
    }

    job_id = _train_manager.start_training(current_app._get_current_object(), dataset_id, params)
    return jsonify({'job_id': job_id, 'dataset_id': dataset_id})


@bp.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    if dataset_id is None:
        return jsonify({'error': 'dataset_id is required'}), 400
    try:
        dataset_id = int(dataset_id)
    except Exception:
        return jsonify({'error': 'dataset_id must be an integer'}), 400

    base_model = str(data.get('base_model', 'yolov8s.pt')).strip()
    if base_model not in YOLO_BASE_MODELS:
        return jsonify({'error': f'base_model must be one of: {", ".join(YOLO_BASE_MODELS)}'}), 400

    device = str(data.get('device', '')).strip()
    try:
        payload = recommend_training_params(
            dataset_id=dataset_id,
            base_model=base_model,
            device=device,
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Could not generate recommendations: {exc}'}), 500

    return jsonify(payload)


@bp.route('/status/<job_id>')
def status(job_id):
    payload = _train_manager.get_status(job_id)
    if not payload:
        return jsonify({'error': 'job not found'}), 404
    return jsonify(payload)


@bp.route('/models')
def models():
    rows = list_trained_models()
    return jsonify([
        {
            'id': item.id,
            'name': item.name,
            'pipeline_slug': item.pipeline_slug,
            'dataset_id': item.dataset_id,
            'model_path': item.model_path,
            'metrics': item.metrics,
            'is_active': item.is_active,
            'created_at': item.created_at.isoformat() if item.created_at else '',
        }
        for item in rows
    ])


@bp.route('/models/<int:model_id>/activate', methods=['POST'])
def activate(model_id):
    model = activate_trained_model(model_id)
    return jsonify({
        'ok': True,
        'id': model.id,
        'name': model.name,
        'is_active': model.is_active,
    })
