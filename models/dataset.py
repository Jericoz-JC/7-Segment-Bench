from datetime import datetime, timezone
from models import db


class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True)
    description = db.Column(db.Text, default='')
    lighting_tag = db.Column(db.String(100), default='')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    images = db.relationship('Image', backref='dataset', lazy='dynamic',
                             cascade='all, delete-orphan')
    benchmark_runs = db.relationship('BenchmarkRun', backref='dataset', lazy='dynamic')

    @property
    def image_count(self):
        return self.images.count()

    @property
    def labeled_count(self):
        from models.label import Label
        return self.images.join(Label).distinct().count()
