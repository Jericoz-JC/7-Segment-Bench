import csv
import io
import json
from flask import Blueprint, request, jsonify, Response
from models.benchmark import BenchmarkRun, BenchmarkResult
from services.label_service import benchmark_target, normalize_and_dedupe_dataset_labels

bp = Blueprint('export', __name__)


@bp.route('/csv/<int:run_id>')
def export_csv(run_id):
    run = BenchmarkRun.query.get_or_404(run_id)
    results = BenchmarkResult.query.filter_by(run_id=run_id).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'image_id', 'pipeline', 'predicted', 'ground_truth',
        'is_correct', 'char_correct', 'char_total', 'latency_ms',
        'confidence', 'error_message'
    ])
    for r in results:
        writer.writerow([
            r.image_id, r.pipeline_slug, r.predicted, r.ground_truth,
            r.is_correct, r.char_correct, r.char_total,
            round(r.latency_ms, 2), round(r.confidence, 4), r.error_message
        ])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=benchmark_{run_id}.csv'}
    )


@bp.route('/json/<int:run_id>')
def export_json(run_id):
    run = BenchmarkRun.query.get_or_404(run_id)
    results = BenchmarkResult.query.filter_by(run_id=run_id).all()

    data = {
        'run': {
            'id': run.id, 'name': run.name, 'status': run.status,
            'pipeline_slugs': run.pipeline_slugs,
            'created_at': run.created_at.isoformat(),
        },
        'results': [{
            'image_id': r.image_id,
            'pipeline': r.pipeline_slug,
            'predicted': r.predicted,
            'ground_truth': r.ground_truth,
            'is_correct': r.is_correct,
            'char_correct': r.char_correct,
            'char_total': r.char_total,
            'latency_ms': round(r.latency_ms, 2),
            'confidence': round(r.confidence, 4),
            'error': r.error_message,
        } for r in results]
    }

    return Response(
        json.dumps(data, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename=benchmark_{run_id}.json'}
    )


@bp.route('/labels/<int:dataset_id>')
def export_labels(dataset_id):
    """Export labels for a dataset as JSON (for backup/reimport)."""
    from models.dataset import Dataset
    from models.image import Image
    from models.label import Label

    dataset = Dataset.query.get_or_404(dataset_id)
    normalize_and_dedupe_dataset_labels(dataset.id)
    images = dataset.images.order_by(Image.filename).all()
    labels_data = []
    for img in images:
        labels = img.labels.order_by(Label.updated_at.desc(), Label.id.desc()).all()
        for lbl in labels:
            labels_data.append({
                'filename': img.filename,
                'roi_x': lbl.roi_x, 'roi_y': lbl.roi_y,
                'roi_width': lbl.roi_width, 'roi_height': lbl.roi_height,
                'ground_truth': lbl.ground_truth,
                'ground_truth_raw': lbl.ground_truth,
                'benchmark_target': benchmark_target(lbl.ground_truth),
                'display_type': lbl.display_type,
                'num_digits': lbl.num_digits,
                'label_source': lbl.labeled_by,
                'verified': str(lbl.labeled_by or '').lower() == 'manual',
                'updated_at': lbl.updated_at.isoformat() if lbl.updated_at else None,
            })

    return Response(
        json.dumps(labels_data, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename=labels_{dataset.name}.json'}
    )
