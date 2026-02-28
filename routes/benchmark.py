from flask import Blueprint, render_template, request, jsonify, Response
from models import db
from models.dataset import Dataset
from models.benchmark import BenchmarkRun
from services.benchmark_runner import BenchmarkRunner
import pipelines

bp = Blueprint('benchmark', __name__)

# Global runner reference for SSE
_runner: BenchmarkRunner | None = None


@bp.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    available_pipelines = pipelines.list_pipelines()
    recent_runs = BenchmarkRun.query.order_by(BenchmarkRun.created_at.desc()).limit(10).all()
    return render_template('benchmark.html', datasets=datasets,
                           pipelines=available_pipelines, recent_runs=recent_runs)


@bp.route('/start', methods=['POST'])
def start_benchmark():
    global _runner
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    pipeline_slugs = data.get('pipeline_slugs', [])
    run_name = data.get('name', 'Benchmark Run')

    if not dataset_id or not pipeline_slugs:
        return jsonify({'error': 'Dataset and at least one pipeline required'}), 400

    dataset = Dataset.query.get_or_404(dataset_id)

    run = BenchmarkRun(
        name=run_name,
        dataset_id=dataset_id,
        status='pending',
    )
    run.pipeline_slugs = pipeline_slugs
    run.pipeline_configs = data.get('pipeline_configs', {})
    db.session.add(run)
    db.session.commit()

    _runner = BenchmarkRunner(run.id)
    _runner.start()

    return jsonify({'run_id': run.id})


@bp.route('/progress/<int:run_id>')
def progress(run_id):
    """SSE endpoint for benchmark progress."""
    def event_stream():
        run = BenchmarkRun.query.get(run_id)
        if not run:
            yield f'data: {{"error": "Run not found"}}\n\n'
            return

        global _runner
        if _runner and _runner.run_id == run_id:
            for event in _runner.events():
                yield f'data: {event}\n\n'
        else:
            yield f'data: {{"status": "{run.status}", "processed": {run.processed}, "total": {run.total}}}\n\n'

    return Response(event_stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@bp.route('/status/<int:run_id>')
def status(run_id):
    run = BenchmarkRun.query.get_or_404(run_id)
    return jsonify({
        'id': run.id,
        'status': run.status,
        'processed': run.processed,
        'total': run.total,
        'progress_pct': run.progress_pct,
    })
