"""Validation helpers for pipeline runtime checks."""

from __future__ import annotations

import subprocess
from services.tesseract_runtime import resolve_tesseract_cmd


def get_tesseract_version_info() -> dict:
    """Return runtime status for `tesseract --version`."""
    cmd = resolve_tesseract_cmd()
    if not cmd:
        return {
            'ok': False,
            'version': '',
            'error': 'tesseract executable not found in PATH',
        }

    try:
        proc = subprocess.run(
            [cmd, '--version'],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return {
            'ok': False,
            'version': '',
            'error': 'tesseract executable not found in PATH',
        }
    except Exception as exc:
        return {
            'ok': False,
            'version': '',
            'error': str(exc),
        }

    output = (proc.stdout or proc.stderr or '').strip()
    first_line = output.splitlines()[0] if output else ''

    if proc.returncode != 0:
        return {
            'ok': False,
            'version': '',
            'error': first_line or f'tesseract returned exit code {proc.returncode}',
        }

    return {
        'ok': True,
        'version': first_line,
        'error': '',
    }
