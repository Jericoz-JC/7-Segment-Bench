import cv2
from flask import Blueprint, render_template, request, jsonify
import pipelines
from models import db
from models.dataset import Dataset
from models.draft_annotation import DraftAnnotation
from models.image import Image
from models.label import Label
from pipelines.base import ROI
from services.image_service import get_image_path
from services.draft_annotation_service import serialize_draft_annotation
from services.label_service import (
    benchmark_target,
    import_labels_csv,
    import_labels_json,
    normalize_and_dedupe_dataset_labels,
    normalize_ground_truth,
    sanitize_ground_truth_raw,
)

bp = Blueprint('label', __name__)


@bp.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template('label_select.html', datasets=datasets)


@bp.route('/dataset/<int:dataset_id>')
def label_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    normalize_and_dedupe_dataset_labels(dataset.id)
    images = dataset.images.order_by(Image.filename).all()
    rows = (
        db.session.query(Label.image_id, Label.labeled_by)
        .join(Image)
        .filter(Image.dataset_id == dataset.id)
        .all()
    )
    label_status = {}
    for image_id, labeled_by in rows:
        source = str(labeled_by or '').lower()
        status = 'verified' if source == 'manual' else 'suggested'
        # verified overrides non-verified if both exist
        if label_status.get(image_id) != 'verified':
            label_status[image_id] = status

    return render_template(
        'labeler.html',
        dataset=dataset,
        images=images,
        label_status=label_status,
    )


@bp.route('/save', methods=['POST'])
def save_label():
    data = request.get_json() or {}
    image_id = data.get('image_id')
    image = Image.query.get_or_404(image_id)

    try:
        ground_truth_raw, target = normalize_ground_truth(data.get('ground_truth', ''))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    replace = data.get('replace')
    if replace is None:
        replace = True
    elif isinstance(replace, str):
        replace = replace.strip().lower() not in {'0', 'false', 'no', 'off'}
    else:
        replace = bool(replace)

    # Keep one active label per image by default.
    if replace:
        Label.query.filter_by(image_id=image_id).delete()

    display_type = str(data.get('display_type', 'led') or 'led').strip().lower()
    if display_type not in {'led', 'lcd'}:
        display_type = 'led'

    label = Label(
        image_id=image_id,
        roi_x=int(data.get('roi_x', 0)),
        roi_y=int(data.get('roi_y', 0)),
        roi_width=int(data.get('roi_width', image.width)),
        roi_height=int(data.get('roi_height', image.height)),
        ground_truth=ground_truth_raw,
        num_digits=len(target),
        display_type=display_type,
        labeled_by='manual',
    )
    db.session.add(label)
    db.session.commit()
    return jsonify({
        'id': label.id,
        'ground_truth': label.ground_truth,
        'ground_truth_raw': label.ground_truth,
        'benchmark_target': target,
        'verified': True,
        'label_source': label.labeled_by,
    })


@bp.route('/delete/<int:label_id>', methods=['DELETE'])
def delete_label(label_id):
    label = Label.query.get_or_404(label_id)
    db.session.delete(label)
    db.session.commit()
    return jsonify({'ok': True})


@bp.route('/import', methods=['POST'])
def import_labels():
    dataset_id = request.form.get('dataset_id', type=int)
    if not dataset_id:
        return jsonify({'error': 'Dataset ID required'}), 400
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'error': f'Dataset {dataset_id} not found'}), 404

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = str(file.filename or '').lower()
    if not filename:
        return jsonify({'error': 'Filename is required'}), 400

    try:
        content = file.read().decode('utf-8')
    except UnicodeDecodeError:
        return jsonify({'error': 'Label file must be UTF-8 text'}), 400

    try:
        if filename.endswith('.csv'):
            count = import_labels_csv(dataset_id, content)
        elif filename.endswith('.json'):
            count = import_labels_json(dataset_id, content)
        else:
            return jsonify({'error': 'Only CSV and JSON supported'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid label file: {e}'}), 400

    backfill = normalize_and_dedupe_dataset_labels(dataset.id)
    return jsonify({'imported': count, **backfill})


@bp.route('/image/<int:image_id>/labels')
def get_labels(image_id):
    labels = Label.query.filter_by(image_id=image_id).all()
    labels.sort(
        key=lambda l: (
            str(l.labeled_by or '').lower() == 'manual',
            (l.updated_at or l.created_at).isoformat() if (l.updated_at or l.created_at) else '',
            int(l.id or 0),
        ),
        reverse=True,
    )
    return jsonify([{
        'id': l.id,
        'roi_x': l.roi_x, 'roi_y': l.roi_y,
        'roi_width': l.roi_width, 'roi_height': l.roi_height,
        'ground_truth': l.ground_truth,
        'ground_truth_raw': l.ground_truth,
        'benchmark_target': benchmark_target(l.ground_truth),
        'display_type': l.display_type,
        'num_digits': l.num_digits,
        'label_source': l.labeled_by,
        'verified': str(l.labeled_by or '').lower() == 'manual',
        'updated_at': l.updated_at.isoformat() if l.updated_at else None,
    } for l in labels])


@bp.route('/image/<int:image_id>/draft')
def get_draft(image_id):
    Image.query.get_or_404(image_id)
    draft = DraftAnnotation.query.filter_by(image_id=image_id).first()
    return jsonify(serialize_draft_annotation(draft))


@bp.route('/suggest', methods=['POST'])
def suggest_label():
    data = request.get_json() or {}
    image_id = data.get('image_id')
    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400

    image = Image.query.get_or_404(int(image_id))
    image_data = cv2.imread(get_image_path(image))
    if image_data is None:
        return jsonify({'error': f'Could not read image: {image.filepath}'}), 400

    def _to_int(value, default):
        try:
            return int(value)
        except Exception:
            return default

    h, w = image_data.shape[:2]
    x = _to_int(data.get('roi_x', 0), 0)
    y = _to_int(data.get('roi_y', 0), 0)
    rw = _to_int(data.get('roi_width', w), w)
    rh = _to_int(data.get('roi_height', h), h)

    x = max(0, min(x, max(0, w - 1)))
    y = max(0, min(y, max(0, h - 1)))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    roi = ROI(x=x, y=y, width=rw, height=rh)

    pipe = None
    try:
        pipe = pipelines.get_pipeline('p05_tesseract_ocr', {})
        pipe.load()
        result = pipe.predict_timed(image_data, roi)
    except Exception as e:
        return jsonify({'error': f'Suggestion failed: {e}'}), 400
    finally:
        if pipe is not None:
            try:
                pipe.unload()
            except Exception:
                pass

    if result.error:
        return jsonify({'error': result.error}), 400

    suggested_raw = sanitize_ground_truth_raw(result.predicted)
    target = benchmark_target(suggested_raw)
    if not target:
        return jsonify({
            'suggested_raw': '',
            'benchmark_target': '',
            'confidence': round(float(result.confidence or 0.0), 4),
            'source': 'p05_tesseract_ocr',
            'error': 'No digits detected',
        })

    return jsonify({
        'suggested_raw': suggested_raw,
        'benchmark_target': target,
        'confidence': round(float(result.confidence or 0.0), 4),
        'source': 'p05_tesseract_ocr',
    })
