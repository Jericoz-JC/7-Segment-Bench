"""Helpers for resolving the local Tesseract executable path."""

from __future__ import annotations

import os
import shutil
from typing import Iterable


def _candidate_paths() -> Iterable[str]:
    env_cmd = str(os.environ.get('TESSERACT_CMD', '') or '').strip()
    if env_cmd:
        yield env_cmd

    which_cmd = shutil.which('tesseract')
    if which_cmd:
        yield which_cmd

    # Common Windows install locations.
    yield r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    yield r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def resolve_tesseract_cmd() -> str | None:
    """Return a usable tesseract executable path, or None if unresolved."""
    for path in _candidate_paths():
        if not path:
            continue
        expanded = os.path.expandvars(path)
        if os.path.isfile(expanded):
            return expanded
    return None

