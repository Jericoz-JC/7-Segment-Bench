"""Draft annotation helpers for unlabeled upload batches."""

from models import db
from models.draft_annotation import DraftAnnotation
from models.image import Image


def normalize_display_type(value: str) -> str:
    display_type = str(value or 'led').strip().lower()
    if display_type not in {'led', 'lcd'}:
        raise ValueError('display_type must be either led or lcd')
    return display_type


def upsert_draft_annotation(image: Image, display_type: str) -> DraftAnnotation:
    draft = DraftAnnotation.query.filter_by(image_id=image.id).first()
    if draft is None:
        draft = DraftAnnotation(image_id=image.id)
        db.session.add(draft)

    draft.roi_x = 0
    draft.roi_y = 0
    draft.roi_width = int(image.width or 0)
    draft.roi_height = int(image.height or 0)
    draft.display_type = normalize_display_type(display_type)
    db.session.commit()
    return draft


def serialize_draft_annotation(draft: DraftAnnotation | None) -> dict | None:
    if draft is None:
        return None
    return {
        'id': draft.id,
        'image_id': draft.image_id,
        'roi_x': draft.roi_x,
        'roi_y': draft.roi_y,
        'roi_width': draft.roi_width,
        'roi_height': draft.roi_height,
        'display_type': draft.display_type,
        'updated_at': draft.updated_at.isoformat() if draft.updated_at else None,
    }


def serialize_manifest_entry(image: Image, draft: DraftAnnotation) -> dict:
    return {
        'filename': image.filename,
        'roi_x': draft.roi_x,
        'roi_y': draft.roi_y,
        'roi_width': draft.roi_width,
        'roi_height': draft.roi_height,
        'display_type': draft.display_type,
    }
