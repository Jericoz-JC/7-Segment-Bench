import os
from flask import Blueprint, render_template, request, jsonify, current_app
from models import db
from models.dataset import Dataset
from models.image import Image
from services.image_service import save_uploaded_image

bp = Blueprint('upload', __name__)


@bp.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template('upload.html', datasets=datasets)


@bp.route('/dataset', methods=['POST'])
def create_dataset():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Dataset name is required'}), 400
    if Dataset.query.filter_by(name=name).first():
        return jsonify({'error': 'Dataset already exists'}), 400
    ds = Dataset(
        name=name,
        description=request.form.get('description', ''),
        lighting_tag=request.form.get('lighting_tag', ''),
    )
    db.session.add(ds)
    db.session.commit()
    return jsonify({'id': ds.id, 'name': ds.name})


@bp.route('/images', methods=['POST'])
def upload_images():
    dataset_id = request.form.get('dataset_id', type=int)
    if not dataset_id:
        return jsonify({'error': 'Dataset ID required'}), 400
    dataset = Dataset.query.get_or_404(dataset_id)

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    allowed = current_app.config['ALLOWED_EXTENSIONS']
    results = []
    for f in files:
        ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
        if ext not in allowed:
            results.append({'filename': f.filename, 'error': 'Invalid file type'})
            continue
        try:
            img_record = save_uploaded_image(f, dataset)
            results.append({'filename': img_record.filename, 'id': img_record.id})
        except Exception as e:
            results.append({'filename': f.filename, 'error': str(e)})

    return jsonify({'uploaded': results})
