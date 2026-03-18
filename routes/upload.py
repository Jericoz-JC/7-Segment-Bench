from flask import Blueprint, render_template, request, jsonify, current_app, url_for
from models import db
from models.dataset import Dataset
from services.image_service import save_uploaded_image
from services.draft_annotation_service import (
    normalize_display_type,
    serialize_manifest_entry,
    upsert_draft_annotation,
)
from services.external_dataset_catalog import catalog_payload
from services.external_dataset_installer import (
    ExternalDatasetValidationError,
    validate_install_request,
)
from services.external_dataset_jobs import ExternalDatasetInstallManager

bp = Blueprint('upload', __name__)
_install_manager = ExternalDatasetInstallManager()


def _resolve_dataset_from_request():
    dataset_id = request.form.get('dataset_id', type=int)
    raw_name = str(request.form.get('name', '') or '').strip()

    if dataset_id and raw_name:
        raise ValueError('Choose an existing dataset or enter a new dataset name, not both')
    if dataset_id:
        return Dataset.query.get_or_404(dataset_id), False
    if not raw_name:
        raise ValueError('Select an existing dataset or enter a new dataset name')
    if Dataset.query.filter_by(name=raw_name).first():
        raise ValueError('Dataset already exists')

    dataset = Dataset(
        name=raw_name,
        description=request.form.get('description', ''),
        lighting_tag=request.form.get('lighting_tag', ''),
    )
    db.session.add(dataset)
    db.session.commit()
    return dataset, True


@bp.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    external_datasets = catalog_payload()
    return render_template('upload.html', datasets=datasets, external_datasets=external_datasets)


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

    uploaded_count = sum(1 for item in results if not item.get('error'))
    failed_count = len(results) - uploaded_count
    return jsonify({
        'uploaded': results,
        'uploaded_count': uploaded_count,
        'failed_count': failed_count,
    })


@bp.route('/unlabeled-batch', methods=['POST'])
def upload_unlabeled_batch():
    try:
        dataset, created_dataset = _resolve_dataset_from_request()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    raw_display_type = request.form.get('display_type', '')
    if not str(raw_display_type or '').strip():
        return jsonify({'error': 'display_type must be either led or lcd'}), 400
    try:
        display_type = normalize_display_type(raw_display_type)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    allowed = current_app.config['ALLOWED_EXTENSIONS']
    results = []
    manifest = []

    for f in files:
        ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
        if ext not in allowed:
            shown_ext = f'.{ext}' if ext else '(missing extension)'
            results.append({'filename': f.filename, 'error': f'Unsupported file type: {shown_ext}'})
            continue
        try:
            img_record = save_uploaded_image(f, dataset)
            draft = upsert_draft_annotation(img_record, display_type)
            results.append({'filename': img_record.filename, 'id': img_record.id})
            manifest.append(serialize_manifest_entry(img_record, draft))
        except Exception as e:
            results.append({'filename': f.filename, 'error': str(e)})

    uploaded_count = sum(1 for item in results if not item.get('error'))
    failed_count = len(results) - uploaded_count
    manifest_filename = f'{dataset.name}_unlabeled_manifest.json'

    return jsonify({
        'dataset': {
            'id': dataset.id,
            'name': dataset.name,
            'image_count': dataset.image_count,
            'created': created_dataset,
        },
        'uploaded': results,
        'uploaded_count': uploaded_count,
        'failed_count': failed_count,
        'display_type': display_type,
        'manifest': manifest,
        'manifest_filename': manifest_filename,
        'label_url': url_for('label.label_dataset', dataset_id=dataset.id),
    })


@bp.route('/external-datasets')
def list_external_datasets():
    return jsonify(catalog_payload())


@bp.route('/external-datasets/install', methods=['POST'])
def install_external_dataset():
    data = request.get_json(silent=True) or {}
    dataset_key = str(data.get('dataset_key', '')).strip()
    mode = str(data.get('mode', 'quick')).strip().lower()

    try:
        validate_install_request(dataset_key, mode)
    except ExternalDatasetValidationError as e:
        return jsonify({'error': str(e)}), 400

    app_obj = current_app._get_current_object()
    job_id = _install_manager.start_install(app_obj, dataset_key, mode)
    return jsonify({'job_id': job_id})


@bp.route('/external-datasets/status/<job_id>')
def external_dataset_status(job_id):
    status = _install_manager.get_status(job_id)
    if not status:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(status)
