from flask import Blueprint, render_template
from models import db
from models.dataset import Dataset
from models.image import Image
from models.label import Label
from models.benchmark import BenchmarkRun

bp = Blueprint('dashboard', __name__)


@bp.route('/')
def index():
    stats = {
        'datasets': Dataset.query.count(),
        'images': Image.query.count(),
        'labels': Label.query.count(),
        'runs': BenchmarkRun.query.count(),
    }
    recent_runs = BenchmarkRun.query.order_by(BenchmarkRun.created_at.desc()).limit(5).all()
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template('dashboard.html', stats=stats, recent_runs=recent_runs,
                           datasets=datasets)
