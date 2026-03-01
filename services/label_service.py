"""Label import/export and normalization service."""

import csv
import io
import json
import re
from models import db
from models.image import Image
from models.label import Label


_GT_ALLOWED_CHARS = set("0123456789-.")


def sanitize_ground_truth_raw(text: str) -> str:
    """Return compact raw ground truth with only supported symbols."""
    compact = re.sub(r"\s+", "", str(text or "").strip())
    filtered = "".join(ch for ch in compact if ch in _GT_ALLOWED_CHARS)
    return filtered[:50]


def benchmark_target(text: str) -> str:
    """Return digits-only target used for benchmark scoring."""
    raw = sanitize_ground_truth_raw(text)
    return "".join(ch for ch in raw if ch.isdigit())


def normalize_ground_truth(text: str) -> tuple[str, str]:
    """Normalize and validate a ground-truth string.

    Returns:
        (ground_truth_raw, benchmark_target_digits)
    """
    raw = sanitize_ground_truth_raw(text)
    if not raw:
        raise ValueError("Ground truth is required")
    target = benchmark_target(raw)
    if not target:
        raise ValueError("Ground truth must include at least one digit")
    return raw, target


def choose_preferred_label(labels: list[Label]) -> Label:
    """Select a single active label for an image.

    Policy:
      1) Prefer latest manual label.
      2) Otherwise latest label by updated_at/created_at/id.
    """
    if not labels:
        raise ValueError("labels must not be empty")

    def _sort_key(label: Label) -> tuple[str, int]:
        dt = label.updated_at or label.created_at
        # ISO string keeps ordering deterministic across naive/aware datetimes.
        dt_key = dt.isoformat() if dt else ""
        return dt_key, int(label.id or 0)

    manual = [lbl for lbl in labels if (lbl.labeled_by or "").lower() == "manual"]
    pool = manual or labels
    return max(pool, key=_sort_key)


def normalize_and_dedupe_dataset_labels(dataset_id: int) -> dict:
    """Normalize labels and keep one active label per image for a dataset."""
    labels = (
        db.session.query(Label)
        .join(Image)
        .filter(Image.dataset_id == dataset_id)
        .all()
    )
    if not labels:
        return {"normalized": 0, "deleted": 0, "deduped_images": 0}

    dirty = False
    by_image: dict[int, list[Label]] = {}
    delete_map: dict[int, Label] = {}
    normalized_count = 0
    deduped_images = 0

    for lbl in labels:
        raw = sanitize_ground_truth_raw(lbl.ground_truth)
        target = benchmark_target(raw)
        if not target:
            delete_map[lbl.id] = lbl
            continue

        if raw != (lbl.ground_truth or ""):
            lbl.ground_truth = raw
            dirty = True
            normalized_count += 1

        digits = len(target)
        if int(lbl.num_digits or 0) != digits:
            lbl.num_digits = digits
            dirty = True
            normalized_count += 1

        by_image.setdefault(lbl.image_id, []).append(lbl)

    for image_id, image_labels in by_image.items():
        if len(image_labels) <= 1:
            continue
        keep = choose_preferred_label(image_labels)
        deduped_images += 1
        for lbl in image_labels:
            if lbl.id != keep.id:
                delete_map[lbl.id] = lbl

    for lbl in delete_map.values():
        db.session.delete(lbl)
        dirty = True

    if dirty:
        db.session.commit()

    return {
        "normalized": normalized_count,
        "deleted": len(delete_map),
        "deduped_images": deduped_images,
    }


def import_labels_csv(dataset_id: int, content: str) -> int:
    """Import labels from CSV content.

    Expected columns: filename, ground_truth, roi_x, roi_y, roi_width, roi_height,
                      display_type (optional)
    """
    reader = csv.DictReader(io.StringIO(content))
    count = 0
    for row in reader:
        filename = row.get('filename', '').strip()
        raw_gt = row.get('ground_truth', '')
        if not filename or not raw_gt:
            continue

        image = Image.query.filter_by(
            dataset_id=dataset_id, filename=filename
        ).first()
        if not image:
            continue

        try:
            ground_truth, target = normalize_ground_truth(raw_gt)
        except ValueError:
            continue

        label = Label(
            image_id=image.id,
            roi_x=int(row.get('roi_x', 0)),
            roi_y=int(row.get('roi_y', 0)),
            roi_width=int(row.get('roi_width', 0)) or image.width,
            roi_height=int(row.get('roi_height', 0)) or image.height,
            ground_truth=ground_truth,
            num_digits=len(target),
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
        raw_gt = item.get('ground_truth', '')
        if not filename or not raw_gt:
            continue

        image = Image.query.filter_by(
            dataset_id=dataset_id, filename=filename
        ).first()
        if not image:
            continue

        try:
            ground_truth, target = normalize_ground_truth(raw_gt)
        except ValueError:
            continue

        label = Label(
            image_id=image.id,
            roi_x=int(item.get('roi_x', 0)),
            roi_y=int(item.get('roi_y', 0)),
            roi_width=int(item.get('roi_width', 0)) or image.width,
            roi_height=int(item.get('roi_height', 0)) or image.height,
            ground_truth=ground_truth,
            num_digits=len(target),
            display_type=item.get('display_type', 'led'),
            labeled_by='json',
        )
        db.session.add(label)
        count += 1

    db.session.commit()
    return count
