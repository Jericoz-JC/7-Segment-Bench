"""Flask application factory."""

import os
from flask import Flask, send_from_directory
from config import config_map


def create_app(config_name: str | None = None) -> Flask:
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'default')

    app = Flask(__name__)
    app.config.from_object(config_map[config_name])

    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['THUMBNAIL_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.instance_path), exist_ok=True)

    # Initialize database
    from models import db
    db.init_app(app)

    with app.app_context():
        # Import models so tables are created
        import models.dataset
        import models.image
        import models.label
        import models.benchmark
        db.create_all()

    # Register blueprints
    from routes.dashboard import bp as dashboard_bp
    from routes.upload import bp as upload_bp
    from routes.label import bp as label_bp
    from routes.benchmark import bp as benchmark_bp
    from routes.results import bp as results_bp
    from routes.single_test import bp as single_test_bp
    from routes.export import bp as export_bp
    from routes.api import bp as api_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(label_bp, url_prefix='/label')
    app.register_blueprint(benchmark_bp, url_prefix='/benchmark')
    app.register_blueprint(results_bp, url_prefix='/results')
    app.register_blueprint(single_test_bp, url_prefix='/test')
    app.register_blueprint(export_bp, url_prefix='/export')
    app.register_blueprint(api_bp, url_prefix='/api')

    # Serve uploaded images and thumbnails
    @app.route('/data/uploads/<path:filename>')
    def serve_upload(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/data/thumbnails/<path:filename>')
    def serve_thumbnail(filename):
        return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
