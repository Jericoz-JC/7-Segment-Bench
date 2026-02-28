"""Pipeline registry — discovers and manages all CV pipeline implementations."""

from pipelines.base import BasePipeline, PipelineResult, ROI

# Registry: slug -> pipeline class
_REGISTRY: dict[str, type[BasePipeline]] = {}


def register(cls: type[BasePipeline]) -> type[BasePipeline]:
    """Decorator to register a pipeline class."""
    _REGISTRY[cls.slug] = cls
    return cls


def get_pipeline(slug: str, config: dict | None = None) -> BasePipeline:
    """Instantiate a pipeline by slug."""
    if slug not in _REGISTRY:
        raise KeyError(f'Unknown pipeline: {slug}. Available: {list(_REGISTRY.keys())}')
    return _REGISTRY[slug](config)


def list_pipelines() -> list[dict]:
    """Return metadata for all registered pipelines."""
    results = []
    for slug, cls in _REGISTRY.items():
        results.append({
            'slug': cls.slug,
            'name': cls.name,
            'description': cls.description,
            'is_online': cls.is_online,
            'is_trainable': cls.is_trainable,
        })
    return sorted(results, key=lambda p: p['slug'])


def get_registry() -> dict[str, type[BasePipeline]]:
    return dict(_REGISTRY)


# Import all pipeline modules to trigger registration
def _import_pipelines():
    import importlib
    modules = [
        'pipelines.p01_global_threshold',
        'pipelines.p02_adaptive_clahe',
        'pipelines.p03_hsv_color',
        'pipelines.p04_template_matching',
        'pipelines.p05_tesseract_ocr',
        'pipelines.p06_yolo_nano',
        'pipelines.p07_llm_vision',
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass  # Missing dependency — skip pipeline


_import_pipelines()
