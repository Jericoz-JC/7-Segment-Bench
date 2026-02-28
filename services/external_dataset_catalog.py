"""Curated external dataset catalog for quick import in the Upload UI."""

from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CuratedDataset:
    key: str
    name: str
    source_url: str
    license_name: str
    access: str
    notes: str
    raw_cache_dir: str
    quick_count: int
    supports_full: bool = True
    requires_env: str = ""
    requires_modules: tuple[str, ...] = ()
    dataset_name_prefix: str = ""

    @property
    def raw_cache_path(self) -> Path:
        return BASE_DIR / self.raw_cache_dir


CATALOG: tuple[CuratedDataset, ...] = (
    CuratedDataset(
        key="hf_7seg_ocr",
        name="7SEG_OCR (Hugging Face)",
        source_url="https://huggingface.co/datasets/HelloImMrGrey/7SEG_OCR",
        license_name="MIT",
        access="Open",
        notes="Image-to-text labels; includes decimals/signs.",
        raw_cache_dir="external_datasets/staging/hf_7seg_ocr_images",
        quick_count=250,
        supports_full=True,
        requires_modules=("datasets",),
        dataset_name_prefix="hf_7seg_ocr",
    ),
    CuratedDataset(
        key="mendeley_fnn44p4mj8",
        name="Mendeley 7-Segment Energy Meter",
        source_url="https://data.mendeley.com/datasets/fnn44p4mj8/1",
        license_name="CC0 1.0",
        access="Open",
        notes="Real meter images with annotation files.",
        raw_cache_dir="external_datasets/mendeley_fnn44p4mj8_v1",
        quick_count=80,
        supports_full=True,
        dataset_name_prefix="mendeley_fnn44p4mj8",
    ),
    CuratedDataset(
        key="roboflow_seven_segment_digits",
        name="seven-segment-digits (Roboflow)",
        source_url="https://universe.roboflow.com/charlie-srmko/seven-segment-digits-uptcy",
        license_name="CC BY 4.0",
        access="Open (API key required to download)",
        notes="Detection labels with extra symbols/classes.",
        raw_cache_dir="external_datasets/roboflow_seven_segment_digits",
        quick_count=300,
        supports_full=True,
        requires_env="ROBOFLOW_API_KEY",
        requires_modules=("roboflow",),
        dataset_name_prefix="rf_seven_segment_digits",
    ),
    CuratedDataset(
        key="roboflow_seven_segment_display_ocr",
        name="Seven Segment Display OCR (Roboflow)",
        source_url="https://universe.roboflow.com/fyp-zodww/seven-segment-display-ocr-lguqw",
        license_name="CC BY 4.0",
        access="Open (API key required to download)",
        notes="Detection labels for digits and unit symbols.",
        raw_cache_dir="external_datasets/roboflow_seven_segment_display_ocr",
        quick_count=250,
        supports_full=True,
        requires_env="ROBOFLOW_API_KEY",
        requires_modules=("roboflow",),
        dataset_name_prefix="rf_seven_segment_display_ocr",
    ),
)


def list_curated_datasets() -> list[CuratedDataset]:
    return list(CATALOG)


def get_curated_dataset(dataset_key: str) -> CuratedDataset:
    for item in CATALOG:
        if item.key == dataset_key:
            return item
    raise KeyError(f"Unknown dataset key: {dataset_key}")


def is_raw_cache_installed(item: CuratedDataset) -> bool:
    path = item.raw_cache_path
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    try:
        return any(path.iterdir())
    except OSError:
        return False


def check_access(item: CuratedDataset) -> tuple[bool, str]:
    if item.requires_env:
        if os.environ.get(item.requires_env, "").strip():
            pass
        else:
            return False, f"{item.requires_env} is required for this dataset."
    for module_name in item.requires_modules:
        if importlib.util.find_spec(module_name) is None:
            return False, f"Python package '{module_name}' is required for this dataset."
    return True, ""


def catalog_payload() -> list[dict]:
    payload = []
    for item in list_curated_datasets():
        env_present = bool(os.environ.get(item.requires_env, "").strip()) if item.requires_env else True
        modules_present = True
        for module_name in item.requires_modules:
            if importlib.util.find_spec(module_name) is None:
                modules_present = False
                break
        install_allowed, install_reason = check_access(item)
        payload.append(
            {
                "key": item.key,
                "name": item.name,
                "source_url": item.source_url,
                "license": item.license_name,
                "access": item.access,
                "notes": item.notes,
                "raw_cache_dir": item.raw_cache_dir,
                "installed_raw": is_raw_cache_installed(item),
                "quick_count": item.quick_count,
                "supports_full": item.supports_full,
                "requires_env": item.requires_env,
                "requires_modules": list(item.requires_modules),
                "env_present": env_present,
                "modules_present": modules_present,
                "install_allowed": install_allowed,
                "install_reason": install_reason,
            }
        )
    return payload
