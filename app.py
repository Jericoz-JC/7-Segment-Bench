"""Flask application factory."""

import os
from flask import Flask, send_from_directory
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError
from config import config_map

REQUIRED_TABLES = {
    'datasets',
    'images',
    'labels',
    'draft_annotations',
    'benchmark_runs',
    'benchmark_results',
    'api_keys',
    'trained_models',
}


def _ensure_sqlite_db_dir(database_uri: str):
    if not database_uri.startswith('sqlite:///'):
        return
    raw_path = database_uri.replace('sqlite:///', '', 1).strip()
    if not raw_path or raw_path == ':memory:':
        return
    db_dir = os.path.dirname(os.path.abspath(raw_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


def _ensure_schema(db) -> bool:
    # Import models so SQLAlchemy metadata includes every table.
    import models.dataset  # noqa: F401
    import models.image  # noqa: F401
    import models.label  # noqa: F401
    import models.draft_annotation  # noqa: F401
    import models.benchmark  # noqa: F401

    def _create_all_safely():
        try:
            db.create_all()
        except OperationalError as exc:
            # Flask's debug reloader can race parent/child startup against SQLite.
            # If another process created the table moments earlier, continue.
            if 'already exists' not in str(exc).lower():
                raise

    _create_all_safely()
    existing = set(inspect(db.engine).get_table_names())
    missing = REQUIRED_TABLES - existing
    if missing:
        _create_all_safely()
        existing = set(inspect(db.engine).get_table_names())
        missing = REQUIRED_TABLES - existing
    return not missing


def create_app(config_name: str | None = None) -> Flask:
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'default')

    app = Flask(__name__)
    app.config.from_object(config_map[config_name])
    # Re-read DATABASE_URL at runtime in case environment changed after module import.
    runtime_db_url = os.environ.get('DATABASE_URL')
    if runtime_db_url:
        app.config['SQLALCHEMY_DATABASE_URI'] = runtime_db_url

    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['THUMBNAIL_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.instance_path), exist_ok=True)
    _ensure_sqlite_db_dir(app.config['SQLALCHEMY_DATABASE_URI'])

    # Initialize database
    from models import db
    db.init_app(app)
    schema_ready = False

    with app.app_context():
        schema_ready = _ensure_schema(db)

    # Register blueprints
    from routes.dashboard import bp as dashboard_bp
    from routes.upload import bp as upload_bp
    from routes.label import bp as label_bp
    from routes.benchmark import bp as benchmark_bp
    from routes.results import bp as results_bp
    from routes.single_test import bp as single_test_bp
    from routes.export import bp as export_bp
    from routes.api import bp as api_bp
    from routes.train_yolo import bp as train_yolo_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(label_bp, url_prefix='/label')
    app.register_blueprint(benchmark_bp, url_prefix='/benchmark')
    app.register_blueprint(results_bp, url_prefix='/results')
    app.register_blueprint(single_test_bp, url_prefix='/test')
    app.register_blueprint(train_yolo_bp, url_prefix='/train/yolo')
    app.register_blueprint(export_bp, url_prefix='/export')
    app.register_blueprint(api_bp, url_prefix='/api')

    # Serve uploaded images and thumbnails
    @app.route('/data/uploads/<path:filename>')
    def serve_upload(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/data/thumbnails/<path:filename>')
    def serve_thumbnail(filename):
        return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)

    @app.before_request
    def ensure_schema_ready():
        nonlocal schema_ready
        if schema_ready:
            return
        with app.app_context():
            schema_ready = _ensure_schema(db)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
