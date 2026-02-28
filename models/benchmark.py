import json
from datetime import datetime, timezone
from models import db


class BenchmarkRun(db.Model):
    __tablename__ = 'benchmark_runs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    pipeline_slugs_json = db.Column(db.Text, default='[]')
    pipeline_configs_json = db.Column(db.Text, default='{}')
    status = db.Column(db.String(30), default='pending')  # pending, running, completed, failed
    processed = db.Column(db.Integer, default=0)
    total = db.Column(db.Integer, default=0)
    summary_json = db.Column(db.Text, default='{}')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    results = db.relationship('BenchmarkResult', backref='run', lazy='dynamic',
                              cascade='all, delete-orphan')

    @property
    def pipeline_slugs(self):
        return json.loads(self.pipeline_slugs_json)

    @pipeline_slugs.setter
    def pipeline_slugs(self, value):
        self.pipeline_slugs_json = json.dumps(value)

    @property
    def pipeline_configs(self):
        return json.loads(self.pipeline_configs_json)

    @pipeline_configs.setter
    def pipeline_configs(self, value):
        self.pipeline_configs_json = json.dumps(value)

    @property
    def summary(self):
        return json.loads(self.summary_json)

    @summary.setter
    def summary(self, value):
        self.summary_json = json.dumps(value)

    @property
    def progress_pct(self):
        if self.total == 0:
            return 0
        return round(100 * self.processed / self.total, 1)


class BenchmarkResult(db.Model):
    __tablename__ = 'benchmark_results'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey('benchmark_runs.id'), nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'), nullable=False)
    label_id = db.Column(db.Integer, db.ForeignKey('labels.id'), nullable=False)
    pipeline_slug = db.Column(db.String(50), nullable=False)
    predicted = db.Column(db.String(50), default='')
    ground_truth = db.Column(db.String(50), nullable=False)
    is_correct = db.Column(db.Boolean, default=False)
    char_correct = db.Column(db.Integer, default=0)
    char_total = db.Column(db.Integer, default=0)
    latency_ms = db.Column(db.Float, default=0.0)
    confidence = db.Column(db.Float, default=0.0)
    error_message = db.Column(db.Text, default='')
    debug_image_path = db.Column(db.String(500), default='')

    @property
    def char_accuracy(self):
        if self.char_total == 0:
            return 0.0
        return self.char_correct / self.char_total


class ApiKey(db.Model):
    __tablename__ = 'api_keys'

    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(30), nullable=False)  # anthropic/openai/openrouter/ollama
    key_value = db.Column(db.String(200), nullable=False)
    is_active = db.Column(db.Boolean, default=True)


class TrainedModel(db.Model):
    __tablename__ = 'trained_models'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    pipeline_slug = db.Column(db.String(50), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    model_path = db.Column(db.String(500), nullable=False)
    metrics_json = db.Column(db.Text, default='{}')
    is_active = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    @property
    def metrics(self):
        return json.loads(self.metrics_json or '{}')

    @metrics.setter
    def metrics(self, value):
        self.metrics_json = json.dumps(value or {})
