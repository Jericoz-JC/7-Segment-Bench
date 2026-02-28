from flask import Blueprint, render_template, jsonify
from models.benchmark import BenchmarkRun, BenchmarkResult
from services.metrics import compute_run_metrics

bp = Blueprint('results', __name__)


@bp.route('/')
def index():
    runs = BenchmarkRun.query.filter_by(status='completed').order_by(
        BenchmarkRun.created_at.desc()).all()
    return render_template('results_list.html', runs=runs)


@bp.route('/<int:run_id>')
def view_run(run_id):
    run = BenchmarkRun.query.get_or_404(run_id)
    metrics = compute_run_metrics(run_id)
    return render_template('results_detail.html', run=run, metrics=metrics)


@bp.route('/<int:run_id>/data')
def run_data(run_id):
    """JSON endpoint for chart data."""
    metrics = compute_run_metrics(run_id)
    return jsonify(metrics)


@bp.route('/<int:run_id>/errors')
def run_errors(run_id):
    """Error gallery — images where pipeline predictions were wrong."""
    run = BenchmarkRun.query.get_or_404(run_id)
    pipeline_slug = __import__('flask').request.args.get('pipeline', '')
    query = BenchmarkResult.query.filter_by(run_id=run_id, is_correct=False)
    if pipeline_slug:
        query = query.filter_by(pipeline_slug=pipeline_slug)
    errors = query.limit(100).all()
    return render_template('error_gallery.html', run=run, errors=errors,
                           pipeline_slug=pipeline_slug)
