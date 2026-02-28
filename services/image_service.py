"""Image storage and thumbnail generation."""

import os
import uuid
from PIL import Image as PILImage
from flask import current_app
from models import db
from models.image import Image
from models.dataset import Dataset


def save_uploaded_image(file, dataset: Dataset) -> Image:
    """Save uploaded file to disk and create DB record."""
    upload_dir = current_app.config['UPLOAD_FOLDER']
    thumb_dir = current_app.config['THUMBNAIL_FOLDER']

    # Generate unique filename
    ext = file.filename.rsplit('.', 1)[-1].lower()
    unique_name = f'{uuid.uuid4().hex}.{ext}'
    filepath = os.path.join(upload_dir, unique_name)

    file.save(filepath)

    # Get dimensions
    with PILImage.open(filepath) as img:
        width, height = img.size

        # Generate thumbnail
        thumb_name = f'thumb_{unique_name}'
        thumb_path = os.path.join(thumb_dir, thumb_name)
        thumb_size = current_app.config['THUMBNAIL_SIZE']
        img_copy = img.copy()
        img_copy.thumbnail(thumb_size, PILImage.LANCZOS)
        img_copy.save(thumb_path)

    record = Image(
        dataset_id=dataset.id,
        filename=file.filename,
        filepath=unique_name,
        thumbnail_path=thumb_name,
        width=width,
        height=height,
        lighting_tag=dataset.lighting_tag,
    )
    db.session.add(record)
    db.session.commit()
    return record


def get_image_path(image: Image) -> str:
    """Get full filesystem path to an image."""
    return os.path.join(current_app.config['UPLOAD_FOLDER'], image.filepath)


def get_thumbnail_path(image: Image) -> str:
    """Get full filesystem path to a thumbnail."""
    return os.path.join(current_app.config['THUMBNAIL_FOLDER'], image.thumbnail_path)
