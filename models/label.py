from datetime import datetime, timezone
from models import db


class Label(db.Model):
    __tablename__ = 'labels'

    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'), nullable=False)
    roi_x = db.Column(db.Integer, nullable=False, default=0)
    roi_y = db.Column(db.Integer, nullable=False, default=0)
    roi_width = db.Column(db.Integer, nullable=False, default=0)
    roi_height = db.Column(db.Integer, nullable=False, default=0)
    ground_truth = db.Column(db.String(50), nullable=False)
    num_digits = db.Column(db.Integer, default=0)
    display_type = db.Column(db.String(20), default='led')  # led or lcd
    labeled_by = db.Column(db.String(20), default='manual')  # manual, csv, json
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    benchmark_results = db.relationship('BenchmarkResult', backref='label', lazy='dynamic')

    @property
    def roi(self):
        return {
            'x': self.roi_x, 'y': self.roi_y,
            'width': self.roi_width, 'height': self.roi_height
        }
