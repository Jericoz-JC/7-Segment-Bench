import base64
import random
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Blueprint, render_template, request, jsonify

import pipelines
from models import db
from models.benchmark import BenchmarkRun
from models.dataset import Dataset
from models.image import Image
from models.label import Label
from pipelines.base import ROI
from routes.benchmark import launch_runner

bp = Blueprint('single_test', __name__)


@bp.route('/')
def index():
    available = pipelines.list_pipelines()
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    labeled_datasets = [d for d in datasets if d.labeled_count > 0]
    return render_template(
        'single_test.html',
        pipelines=available,
        datasets=labeled_datasets,
    )


@bp.route('/run', methods=['POST'])
def run_test():
    """Run all (or selected) pipelines on a single uploaded image."""
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read image into memory
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Parse ROI
    roi_data = request.form.get('roi')
    if roi_data:
        import json
        roi_dict = json.loads(roi_data)
        roi = ROI.from_dict(roi_dict)
    else:
        roi = ROI.full_image(image)

    # Which pipelines to run
    selected = request.form.getlist('pipelines')
    if not selected:
        selected = [p['slug'] for p in pipelines.list_pipelines()]

    results = {}
    for slug in selected:
        try:
            pipe = pipelines.get_pipeline(slug)
            pipe.load()
            result = pipe.predict_timed(image, roi)
            pipe.unload()

            # Encode debug images as base64
            debug_b64 = {}
            for name, dbg_img in result.debug_images.items():
                _, buf = cv2.imencode('.png', dbg_img)
                debug_b64[name] = base64.b64encode(buf).decode('utf-8')

            results[slug] = {
                'predicted': result.predicted,
                'confidence': round(result.confidence, 4),
                'latency_ms': round(result.latency_ms, 2),
                'error': result.error,
                'debug_images': debug_b64,
            }
        except Exception as e:
            results[slug] = {
                'predicted': '',
                'error': str(e),
                'latency_ms': 0,
                'confidence': 0,
                'debug_images': {},
            }

    return jsonify(results)


@bp.route('/run-batch', methods=['POST'])
def run_batch_test():
    """Run selected pipelines on a sampled labeled subset and persist as BenchmarkRun."""
    data = request.get_json() or {}

    dataset_id = data.get('dataset_id')
    pipeline_slugs = data.get('pipeline_slugs', [])
    sample_size = data.get('sample_size', 20)
    run_name = str(data.get('name', '')).strip()
    pipeline_configs = data.get('pipeline_configs', {})

    try:
        dataset_id = int(dataset_id)
    except Exception:
        return jsonify({'error': 'dataset_id is required'}), 400

    try:
        sample_size = int(sample_size)
    except Exception:
        return jsonify({'error': 'sample_size must be an integer'}), 400

    if sample_size < 10 or sample_size > 50:
        return jsonify({'error': 'sample_size must be between 10 and 50'}), 400

    if not isinstance(pipeline_slugs, list) or not pipeline_slugs:
        return jsonify({'error': 'pipeline_slugs must be a non-empty list'}), 400

    valid_slugs = {p['slug'] for p in pipelines.list_pipelines()}
    unknown = [s for s in pipeline_slugs if s not in valid_slugs]
    if unknown:
        return jsonify({'error': f'Unknown pipeline slug(s): {unknown}'}), 400

    if not isinstance(pipeline_configs, dict):
        return jsonify({'error': 'pipeline_configs must be an object'}), 400

    dataset = Dataset.query.get_or_404(dataset_id)

    rows = (
        db.session.query(Image.id)
        .join(Label)
        .filter(Image.dataset_id == dataset.id)
        .distinct()
        .all()
    )
    labeled_image_ids = [r[0] for r in rows]
    if not labeled_image_ids:
        return jsonify({'error': 'Selected dataset has no labeled images'}), 400

    selected_count = min(sample_size, len(labeled_image_ids))
    if len(labeled_image_ids) > selected_count:
        selected_image_ids = random.sample(labeled_image_ids, selected_count)
    else:
        selected_image_ids = list(labeled_image_ids)

    if not run_name:
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        run_name = f'Quick Batch {dataset.name} {ts}'

    run = BenchmarkRun(
        name=run_name,
        dataset_id=dataset.id,
        status='pending',
    )
    run.pipeline_slugs = pipeline_slugs

    merged_configs = dict(pipeline_configs)
    merged_configs['__selection'] = {
        'image_ids': selected_image_ids,
        'mode': 'quick_batch',
        'sample_size': sample_size,
        'selected_count': selected_count,
        'strategy': 'random',
    }
    run.pipeline_configs = merged_configs

    db.session.add(run)
    db.session.commit()

    launch_runner(run.id)

    return jsonify({
        'run_id': run.id,
        'selected_count': selected_count,
        'selected_image_ids': selected_image_ids,
    })
