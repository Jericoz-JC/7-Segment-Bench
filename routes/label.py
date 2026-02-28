import json
import csv
import io
from flask import Blueprint, render_template, request, jsonify
from models import db
from models.dataset import Dataset
from models.image import Image
from models.label import Label
from services.label_service import import_labels_csv, import_labels_json

bp = Blueprint('label', __name__)


@bp.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template('label_select.html', datasets=datasets)


@bp.route('/dataset/<int:dataset_id>')
def label_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    images = dataset.images.order_by(Image.filename).all()
    return render_template('labeler.html', dataset=dataset, images=images)


@bp.route('/save', methods=['POST'])
def save_label():
    data = request.get_json()
    image_id = data.get('image_id')
    image = Image.query.get_or_404(image_id)

    # Delete existing labels for this image if replacing
    if data.get('replace', False):
        Label.query.filter_by(image_id=image_id).delete()

    label = Label(
        image_id=image_id,
        roi_x=int(data.get('roi_x', 0)),
        roi_y=int(data.get('roi_y', 0)),
        roi_width=int(data.get('roi_width', image.width)),
        roi_height=int(data.get('roi_height', image.height)),
        ground_truth=data.get('ground_truth', ''),
        num_digits=len(data.get('ground_truth', '')),
        display_type=data.get('display_type', 'led'),
        labeled_by='manual',
    )
    db.session.add(label)
    db.session.commit()
    return jsonify({'id': label.id, 'ground_truth': label.ground_truth})


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

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = file.filename.lower()
    content = file.read().decode('utf-8')

    if filename.endswith('.csv'):
        count = import_labels_csv(dataset_id, content)
    elif filename.endswith('.json'):
        count = import_labels_json(dataset_id, content)
    else:
        return jsonify({'error': 'Only CSV and JSON supported'}), 400

    return jsonify({'imported': count})


@bp.route('/image/<int:image_id>/labels')
def get_labels(image_id):
    labels = Label.query.filter_by(image_id=image_id).all()
    return jsonify([{
        'id': l.id,
        'roi_x': l.roi_x, 'roi_y': l.roi_y,
        'roi_width': l.roi_width, 'roi_height': l.roi_height,
        'ground_truth': l.ground_truth,
        'display_type': l.display_type,
        'num_digits': l.num_digits,
    } for l in labels])
