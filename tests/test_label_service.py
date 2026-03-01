from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from services.label_service import (
    benchmark_target,
    choose_preferred_label,
    normalize_ground_truth,
    sanitize_ground_truth_raw,
)


def test_sanitize_ground_truth_raw_filters_and_trims():
    assert sanitize_ground_truth_raw(' .- 38a.20 V ') == '.-38.20'


def test_benchmark_target_digits_only():
    assert benchmark_target('.-3820') == '3820'
    assert benchmark_target('-94.06') == '9406'


def test_normalize_ground_truth_requires_digit():
    with pytest.raises(ValueError):
        normalize_ground_truth('-.')


def test_choose_preferred_label_prefers_latest_manual():
    older_manual = SimpleNamespace(
        id=1,
        labeled_by='manual',
        updated_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        created_at=None,
    )
    latest_manual = SimpleNamespace(
        id=2,
        labeled_by='manual',
        updated_at=datetime(2026, 2, 2, tzinfo=timezone.utc),
        created_at=None,
    )
    latest_imported = SimpleNamespace(
        id=3,
        labeled_by='json',
        updated_at=datetime(2026, 2, 3, tzinfo=timezone.utc),
        created_at=None,
    )

    chosen = choose_preferred_label([older_manual, latest_imported, latest_manual])
    assert chosen.id == 2
