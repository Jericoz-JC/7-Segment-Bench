from flask import Blueprint, request, jsonify
from models import db
from models.benchmark import ApiKey

bp = Blueprint('api', __name__)


@bp.route('/keys', methods=['GET'])
def list_keys():
    keys = ApiKey.query.all()
    return jsonify([{
        'id': k.id,
        'provider': k.provider,
        'key_preview': k.key_value[:8] + '...' if len(k.key_value) > 8 else '***',
        'is_active': k.is_active,
    } for k in keys])


@bp.route('/keys', methods=['POST'])
def add_key():
    data = request.get_json()
    provider = data.get('provider', '').strip()
    key_value = data.get('key_value', '').strip()

    if provider not in ('anthropic', 'openai'):
        return jsonify({'error': 'Provider must be anthropic or openai'}), 400
    if not key_value:
        return jsonify({'error': 'API key required'}), 400

    # Deactivate existing keys for this provider
    ApiKey.query.filter_by(provider=provider).update({'is_active': False})

    key = ApiKey(provider=provider, key_value=key_value, is_active=True)
    db.session.add(key)
    db.session.commit()
    return jsonify({'id': key.id, 'provider': key.provider})


@bp.route('/keys/<int:key_id>', methods=['DELETE'])
def delete_key(key_id):
    key = ApiKey.query.get_or_404(key_id)
    db.session.delete(key)
    db.session.commit()
    return jsonify({'ok': True})
