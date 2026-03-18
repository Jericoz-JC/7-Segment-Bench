from datetime import datetime, timezone
from models import db


class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    filename = db.Column(db.String(300), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    thumbnail_path = db.Column(db.String(500), default='')
    width = db.Column(db.Integer, default=0)
    height = db.Column(db.Integer, default=0)
    lighting_tag = db.Column(db.String(100), default='')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    labels = db.relationship('Label', backref='image', lazy='dynamic',
                             cascade='all, delete-orphan')
    draft_annotation = db.relationship(
        'DraftAnnotation',
        backref='image',
        uselist=False,
        cascade='all, delete-orphan',
    )
    benchmark_results = db.relationship('BenchmarkResult', backref='image', lazy='dynamic')
