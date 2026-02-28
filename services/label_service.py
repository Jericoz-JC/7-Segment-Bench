"""Label import/export service."""

import csv
import io
import json
from models import db
from models.image import Image
from models.label import Label


def import_labels_csv(dataset_id: int, content: str) -> int:
    """Import labels from CSV content.

    Expected columns: filename, ground_truth, roi_x, roi_y, roi_width, roi_height,
                      display_type (optional)
    """
    reader = csv.DictReader(io.StringIO(content))
    count = 0
    for row in reader:
        filename = row.get('filename', '').strip()
        ground_truth = row.get('ground_truth', '').strip()
        if not filename or not ground_truth:
            continue

        image = Image.query.filter_by(
            dataset_id=dataset_id, filename=filename
        ).first()
        if not image:
            continue

        label = Label(
            image_id=image.id,
            roi_x=int(row.get('roi_x', 0)),
            roi_y=int(row.get('roi_y', 0)),
            roi_width=int(row.get('roi_width', 0)) or image.width,
            roi_height=int(row.get('roi_height', 0)) or image.height,
            ground_truth=ground_truth,
            num_digits=len(ground_truth),
            display_type=row.get('display_type', 'led').strip() or 'led',
            labeled_by='csv',
        )
        db.session.add(label)
        count += 1

    db.session.commit()
    return count


def import_labels_json(dataset_id: int, content: str) -> int:
    """Import labels from JSON content.

    Expected format: list of objects with same fields as CSV.
    """
    data = json.loads(content)
    if not isinstance(data, list):
        data = [data]

    count = 0
    for item in data:
        filename = item.get('filename', '').strip()
        ground_truth = item.get('ground_truth', '').strip()
        if not filename or not ground_truth:
            continue

        image = Image.query.filter_by(
            dataset_id=dataset_id, filename=filename
        ).first()
        if not image:
            continue

        label = Label(
            image_id=image.id,
            roi_x=int(item.get('roi_x', 0)),
            roi_y=int(item.get('roi_y', 0)),
            roi_width=int(item.get('roi_width', 0)) or image.width,
            roi_height=int(item.get('roi_height', 0)) or image.height,
            ground_truth=ground_truth,
            num_digits=len(ground_truth),
            display_type=item.get('display_type', 'led'),
            labeled_by='json',
        )
        db.session.add(label)
        count += 1

    db.session.commit()
    return count
