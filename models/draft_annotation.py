from datetime import datetime, timezone
from models import db


class DraftAnnotation(db.Model):
    __tablename__ = 'draft_annotations'

    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'), nullable=False, unique=True)
    roi_x = db.Column(db.Integer, nullable=False, default=0)
    roi_y = db.Column(db.Integer, nullable=False, default=0)
    roi_width = db.Column(db.Integer, nullable=False, default=0)
    roi_height = db.Column(db.Integer, nullable=False, default=0)
    display_type = db.Column(db.String(20), default='led')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

