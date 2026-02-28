import os
import base64
import cv2
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app
import pipelines
from pipelines.base import ROI

bp = Blueprint('single_test', __name__)


@bp.route('/')
def index():
    available = pipelines.list_pipelines()
    return render_template('single_test.html', pipelines=available)


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
            results[slug] = {'predicted': '', 'error': str(e), 'latency_ms': 0,
                             'confidence': 0, 'debug_images': {}}

    return jsonify(results)
