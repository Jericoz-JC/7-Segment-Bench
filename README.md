# seven-segment-bench

A Flask app for comparing multiple computer-vision and OCR approaches on 7-segment display readings.

It lets you run classic CV, template methods, OCR, object detection, and LLM vision side-by-side with one shared dataset/labeling/metrics workflow.

## What This Project Does

`seven-segment-bench` helps you:

- build labeled datasets of display images with ROI ground truth
- run multiple pipelines on the same labeled set
- compare string accuracy, character accuracy, latency, and errors
- inspect error cases and confusion trends
- export results for downstream analysis

Primary use case: test how each pipeline holds up across different datasets and image conditions (lighting, glare, angle, blur, color, display type).

## Core Features

- Dataset management:
  - Create named datasets with optional tags (for example: `bright`, `dim`, `glare`).
  - Upload multiple images with thumbnail generation.
- Labeling UI:
  - Draw ROI on canvas.
  - Enter ground-truth digits.
  - Mark display type (`led` or `lcd`).
  - Keyboard workflow for fast annotation.
- Label import/export:
  - Import labels from CSV/JSON.
  - Export dataset labels to JSON.
- Benchmark orchestration:
  - Run selected pipelines over all labeled images.
  - Conditional LLM settings panel for `p07_llm_vision` (provider/model/base URL).
  - Live progress via SSE.
  - Per-image, per-pipeline result persistence.
- Metrics and analytics:
  - String-level accuracy.
  - Character-level accuracy.
  - Latency mean/median/p95.
  - 10x10 digit confusion matrix.
- Result exploration:
  - Comparison table and charts.
  - Error gallery (mispredictions).
  - CSV/JSON benchmark export.
- Quick test mode:
  - Batch mode: select a dataset and run a random 10-50 labeled-image sample.
  - Single-image mode: upload one image with optional ROI for deep debug.
  - Shared LLM settings panel appears automatically when `p07_llm_vision` is selected.
  - LLM provider keys entered in Quick Test are saved as active keys via `/api/keys`.
  - Batch quick tests persist as normal benchmark runs and open in Results.
- API key storage for online models:
  - Store active provider key (`anthropic`, `openai`, `openrouter`, `ollama`) via API routes.
  - `ollama` supports key-optional local use.
- In-app YOLO trainer for `p06`:
  - Launch train/val split export + training job from labeled datasets.
  - Choose base checkpoint (`yolov8n.pt` or `yolov8s.pt`), with `yolov8s.pt` default.
  - Auto-suggest `epochs`, `imgsz`, `batch`, and `seed` from dataset stats.
  - Activate a trained checkpoint as the default `p06` model.

## Built-in Pipelines (7 Total)

All pipelines are auto-registered from `pipelines/__init__.py`.

| Slug | Name | Method | Online | Trainable | Key Config | Notes |
|---|---|---|---|---|---|---|
| `p01_global_threshold` | Global Threshold (Otsu) | Otsu binarization + segment map | No | No | `num_digits` | Fast baseline for clean, high-contrast displays. |
| `p02_adaptive_clahe` | Adaptive + CLAHE | CLAHE + adaptive threshold with Otsu fallback + segment map | No | No | `num_digits`, `clip_limit`, `tile_size`, `block_size`, `c_value` | More robust to non-uniform illumination; includes fallback path for block artifacts. |
| `p03_hsv_color` | HSV Color Filter | HSV mask (red/green/blue) + glare rejection + segment map | No | No | `num_digits`, `color` | Best when segment color is known and stable. |
| `p04_template_matching` | Template Matching | Synthetic template generation + NCC matching | No | No | `num_digits`, `template_height`, `template_width` | Useful non-learning baseline with explicit shape matching. |
| `p05_tesseract_ocr` | Tesseract OCR | OCR with digit whitelist | No | No | `target_height`, `psm`, `tessdata`, `lang` | Requires local Tesseract runtime; sensitive to font and preprocessing. |
| `p06_yolo_nano` | YOLOv8 Detector | Detection model (digit classes) | No | Yes | `model_path`, `ncnn_path` | Training Lab supports `yolov8n.pt` and `yolov8s.pt` base checkpoints; `yolov8s.pt` is the default recommendation. |
| `p07_llm_vision` | LLM Vision API | Zero-shot image-to-digits (Claude/GPT-4o/OpenRouter/Ollama style) | Yes | No | `provider`, `api_key`, `model` | Supports `anthropic`, `openai`, `openrouter`, and local `ollama` providers. |

### How Each Pipeline Works (Mechanics)

- `p01_global_threshold`:
  - ROI crop -> grayscale -> resize -> Otsu threshold -> morphology -> segment fill ratios -> digit lookup.
- `p02_adaptive_clahe`:
  - ROI crop -> grayscale -> resize -> CLAHE enhancement -> adaptive threshold path and Otsu fallback path -> choose stronger-confidence read.
- `p03_hsv_color`:
  - ROI crop -> HSV conversion -> color mask by LED hue -> glare suppression using low-saturation/high-value rejection -> morphology -> segment read.
- `p04_template_matching`:
  - ROI crop -> grayscale/threshold -> digit isolation -> resize synthetic templates per digit -> normalized cross-correlation -> best score per digit.
- `p05_tesseract_ocr`:
  - ROI crop -> grayscale resize -> CLAHE -> threshold -> morphology -> padding -> Tesseract with digit whitelist.
- `p06_yolo_nano`:
  - ROI crop -> YOLO inference -> collect detection boxes/classes/confidence -> sort left-to-right -> concatenate class digits.
- `p07_llm_vision`:
  - ROI crop -> JPEG base64 -> provider API prompt -> parse digit sequence from returned text.

## Quick Test Lab

Quick Test now has two flows in one page:

- **Dataset Batch**:
  - choose a labeled dataset
  - choose sample size (10-50)
  - select pipelines
  - run and stream progress
  - open persisted run in Results
- **Single-Image Debug**:
  - upload image
  - optional ROI draw
  - run selected pipelines and inspect debug images
- **Shared LLM Settings (conditional)**:
  - appears only when `p07_llm_vision` is selected in batch or single pipeline list
  - provider/model/base URL config is sent as `pipeline_configs` for that run
  - entering an API key saves it as the active key for that provider

Batch runs use a random subset and are saved as `BenchmarkRun` records.

## YOLO Training Lab (`p06`)

Use **YOLO Train** in the sidebar to train and activate a `p06` model from app datasets.

Training workflow:

1. Select one labeled dataset.
2. Configure training settings:
   - `base_model` (`yolov8n.pt` or `yolov8s.pt`, default `yolov8s.pt`)
   - `epochs`
   - `imgsz`
   - `batch`
   - `val_ratio`
   - `seed`
   - optional `device`
   - optional auto-suggest button to fill `epochs/imgsz/batch/seed`
3. Start training (background job).
4. Monitor status in the UI.
5. Activate a trained model.

Activated models are stored in `trained_models` and automatically used by `p06` when no explicit `model_path` is passed.

### YOLOv8s Recommendation

Upgrading from `yolov8n.pt` to `yolov8s.pt` is a practical sweet spot for this project:

- around 3x the parameters of `v8n` but still fast for training/inference
- suitable for NCNN export for Raspberry Pi deployment
- with ~3,333 images across 10 digit classes (~333/class), dataset size is strong enough to benefit from `v8s`

The auto-suggest button is heuristic guidance, not hyperparameter search. Final tuning should still be validated against your held-out split.

### Dataset Export Split Note

`export_yolo_dataset(...)` now writes a true train/val split and generates `data.yaml` with:

- `train: train/images`
- `val: val/images`

This avoids train==val leakage and makes validation metrics meaningful.

## Provider Key API

`POST /api/keys`

- `provider`: one of `anthropic`, `openai`, `openrouter`, `ollama`
- `key_value`:
  - required for `anthropic` / `openai` / `openrouter`
  - optional for `ollama` (local endpoint flow)

`GET /api/keys` lists stored providers and active states.

UI note:
- Benchmark and Quick Test call this API when you enter a key in the LLM settings panel.

## End-to-End Workflow

1. Create dataset in `Upload`.
2. Upload images to the dataset.
3. Label images:
   - draw ROI
   - enter `ground_truth`
   - choose `display_type`
4. Optionally import labels from CSV/JSON.
5. Run benchmark:
   - select dataset
   - select pipelines
   - start run and monitor progress
6. Open results:
   - compare pipeline metrics
   - inspect confusion matrix and error gallery
7. Export run outputs (`CSV` or `JSON`) for further analysis.
8. Repeat across additional datasets to test cross-domain generalization.

## Quick Dataset Installs (Upload Page)

The Upload page includes a **Curated External Datasets** card so testers can import ready-to-use datasets without manual conversion.

- `Quick Add`: installs a starter subset for fast smoke testing.
- `Full Add`: installs the full dataset when available.
- Imports run in a background job and show live status in the Upload UI.
- Imports create a new dataset in the app with images and labels ready for Benchmark.

### Curated Datasets

| Key | Source | Access | Local raw cache |
|---|---|---|---|
| `hf_7seg_ocr` | https://huggingface.co/datasets/HelloImMrGrey/7SEG_OCR | Open | `external_datasets/staging/hf_7seg_ocr_images` |
| `mendeley_fnn44p4mj8` | https://data.mendeley.com/datasets/fnn44p4mj8/1 | Open | `external_datasets/mendeley_fnn44p4mj8_v1` |
| `roboflow_seven_segment_digits` | https://universe.roboflow.com/charlie-srmko/seven-segment-digits-uptcy | Requires `ROBOFLOW_API_KEY` | `external_datasets/roboflow_seven_segment_digits` |
| `roboflow_seven_segment_display_ocr` | https://universe.roboflow.com/fyp-zodww/seven-segment-display-ocr-lguqw | Requires `ROBOFLOW_API_KEY` | `external_datasets/roboflow_seven_segment_display_ocr` |

Set Roboflow key when needed:

```powershell
$env:ROBOFLOW_API_KEY = "your_key_here"
```

If a dataset button shows a dependency warning, install missing packages:

```powershell
python -m pip install -r requirements.txt
```

No benchmark is started automatically after import. This keeps testers in control of which pipelines to run.

## Architecture Deep Dive

### High-level System Diagram

```text
+-----------------------------+
| Flask App Factory (app.py)  |
| - config loading            |
| - DB init/create_all        |
| - blueprint registration    |
+--------------+--------------+
               |
     +---------+---------+---------------------+------------------+
     |                   |                     |                  |
+----v----+        +-----v-----+         +-----v-----+      +-----v-----+
| Routes  |        | Services  |         | Pipelines |      | Templates |
| upload  |        | benchmark |         | p01..p07  |      | + JS/CSS  |
| label   |        | metrics   |         | registry  |      | frontend  |
| bench   |        | image IO  |         | base API  |      | UI views  |
| train   |        | yolo jobs |         |           |      |           |
| results |        | label I/O |         +-----+-----+      +-----------+
| export  |        | yolo prep |               |
+----+----+        +-----+-----+               |
     |                   |                     |
     +-------------------+---------------------+
                         |
                   +-----v----------------------+
                   | SQLAlchemy Models          |
                   | Dataset/Image/Label        |
                   | BenchmarkRun/Result/ApiKey |
                   | TrainedModel               |
                   +-----+----------------------+
                         |
                   +-----v----------------------+
                   | SQLite (instance/app.db)   |
                   | data/uploads, thumbnails   |
                   +----------------------------+
```

### Benchmark Execution Flow

1. `POST /benchmark/start` creates `BenchmarkRun`.
2. `BenchmarkRunner` starts background thread.
3. Runner collects labeled images (`Image` + `Label` join).
4. For each selected pipeline slug:
   - instantiate from registry
   - `load()`
   - run `predict_timed()` per labeled image ROI
   - save `BenchmarkResult`
   - emit SSE progress events
   - `unload()`
5. On completion:
   - compute aggregate metrics
   - persist `run.summary`
   - expose in results pages/charts/export APIs

### Pipeline Plugin Pattern

- Common interface from `BasePipeline`:
  - `load()`
  - `predict(image, roi)`
  - `unload()`
  - `predict_timed(...)` wrapper
- `@register` decorator adds class to central registry.
- Registry exposes:
  - `list_pipelines()`
  - `get_pipeline(slug, config)`

This lets the benchmark system treat all methods uniformly.

## Data Model and Storage

### Main Entities

- `Dataset`:
  - name, description, lighting tag
  - links to images and benchmark runs
- `Image`:
  - original filename, stored path, dimensions, thumbnail path
  - belongs to dataset
- `Label`:
  - ROI (`x,y,width,height`)
  - `ground_truth`, `num_digits`, `display_type`
  - belongs to image
- `BenchmarkRun`:
  - selected pipeline slugs/configs
  - status and progress
  - summary metrics JSON
- `BenchmarkResult`:
  - one record per `(run, image, label, pipeline)`
  - prediction, correctness, char metrics, latency, confidence, error
- `ApiKey`:
  - provider key storage and active key flag

### Storage Paths

- Uploaded images: `data/uploads/`
- Thumbnails: `data/thumbnails/`
- SQLite DB: `instance/app.db`

## Setup and Run

## Windows (PowerShell)

```powershell
cd "C:\Users\chris\Documents\CS Projects\Research\seven-segment-bench"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

Optional config:

```powershell
$env:FLASK_CONFIG = "dev"   # or "pi"
python app.py
```

If script execution is blocked in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Linux/macOS

```bash
cd /path/to/seven-segment-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Optional config:

```bash
export FLASK_CONFIG=dev   # or pi
python app.py
```

## Raspberry Pi 5

Use bundled script:

```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

Then:

```bash
source venv/bin/activate
FLASK_CONFIG=pi python app.py
```

## Dataset Discovery for Generalization

The best generalization studies mix:

- open downloadable datasets
- harder academic datasets with field artifacts
- diverse label styles (sequence, detection boxes, symbols, decimal signs)

### Candidate Datasets (checked February 28, 2026)

| Dataset | Link | Access | License | Notes |
|---|---|---|---|---|
| seven-segment-digits (Roboflow) | https://universe.roboflow.com/charlie-srmko/seven-segment-digits-uptcy | Open | CC BY 4.0 | ~4.8k images, detection labels, includes symbols/classes beyond 0-9. |
| Seven segment display OCR (Roboflow) | https://universe.roboflow.com/fyp-zodww/seven-segment-display-ocr-lguqw | Open | CC BY 4.0 | ~1.2k images, classes include 0-9 and unit symbols. |
| 7SEG_OCR (Hugging Face) | https://huggingface.co/datasets/HelloImMrGrey/7SEG_OCR | Open | MIT | ~3.33k rows, image-to-text style labels. |
| Mendeley 7-segment energy meter dataset | https://data.mendeley.com/datasets/fnn44p4mj8/1 | Open | CC0 1.0 | 169 real images with annotation files, useful for small-scale external validity checks. |
| UFPR-AMR | https://web.inf.ufpr.br/vri/databases/ufpr-amr/ | Request (academic) | Research-only | 2,000 annotated meter images; strong academic benchmark lineage. |
| Copel-AMR | https://web.inf.ufpr.br/vri/databases/copel-amr/ | Request (academic) | Academic non-commercial with signed agreement | 12,500 field images, heavy unconstrained artifacts, strong stress test. |

### Dataset Selection Checklist

- Include both LED and LCD where possible.
- Include multiple lighting conditions (dim, bright, glare).
- Include perspective and motion blur cases.
- Include variable digit counts and decimal/sign symbols.
- Confirm labeling quality and class definitions before import.

## Bringing External Datasets Into This App

### Path A: Images only (no labels)

1. Create dataset in Upload page.
2. Upload images.
3. Annotate ROI + ground truth in labeling UI.
4. Run benchmark.

### Path B: Images + existing labels

1. Upload images into a dataset.
2. Convert annotation format into supported import schema.
3. Import labels from Label page (`CSV` or `JSON`).
4. Run benchmark.

### Required Label Import Fields

- `filename` (must match uploaded image original filename)
- `ground_truth`
- `roi_x`
- `roi_y`
- `roi_width`
- `roi_height`
- optional: `display_type` (`led` or `lcd`)

### CSV Example

```csv
filename,ground_truth,roi_x,roi_y,roi_width,roi_height,display_type
meter_001.jpg,5678,120,80,340,120,led
meter_002.jpg,9012,98,74,360,126,lcd
```

### JSON Example

```json
[
  {
    "filename": "meter_003.jpg",
    "ground_truth": "4321",
    "roi_x": 110,
    "roi_y": 70,
    "roi_width": 355,
    "roi_height": 128,
    "display_type": "led"
  }
]
```

### Annotation Conversion Notes

- Detection datasets (per-digit boxes) can be collapsed to one display ROI and ordered digit string.
- If your source dataset has non-digit symbols, decide whether to:
  - keep them and evaluate only compatible pipelines, or
  - normalize labels to digits-only subsets for strict comparability.

## Validation and Test Coverage

## Latest Test Snapshot

Executed on February 28, 2026:

```bash
python -m pytest -q tests
```

Result:

- `50 passed`
- `6 warnings` (legacy SQLAlchemy `Query.get()` + environment-specific pytest cache warnings)

### Unit/Integration Coverage Matrix

| Area | Covered | Notes |
|---|---|---|
| Segment map geometry and classification | Yes | `extract_segment_values`, `classify_digit`, `isolate_digits`, `read_display`. |
| Pipeline `p01_global_threshold` | Yes | Clean and noisy synthetic cases. |
| Pipeline `p02_adaptive_clahe` | Yes | Clean synthetic case. |
| Pipeline `p03_hsv_color` | Yes | Green and red LED synthetic cases. |
| Pipeline `p04_template_matching` | Yes | Clean synthetic case. |
| Metrics utility | Yes | Character accuracy scenarios and length mismatch. |
| ROI dataclass behaviors | Yes | Crop/full-image/from-dict tests. |
| Flask route integration | Yes | Dashboard, upload, label, benchmark, results, single-test, YOLO train, API keys, export labels. |
| Pipeline `p05_tesseract_ocr` | Partial | Runtime preflight endpoint + whitelist behavior checks. |
| Pipeline `p06_yolo_nano` | Partial | Left-to-right detection sorting validation + training/activation route coverage. |
| Pipeline `p07_llm_vision` | Partial | Provider loading checks for OpenRouter and Ollama. |

### Per-Pipeline Validation Status

| Pipeline | Automated Test Status | What is validated today |
|---|---|---|
| `p01_global_threshold` | Covered | Clean image decoding and noisy-image shape/length robustness. |
| `p02_adaptive_clahe` | Covered | Clean synthetic decoding for a full 4-digit case. |
| `p03_hsv_color` | Covered | Green LED and red LED decoding paths. |
| `p04_template_matching` | Covered | Clean synthetic decoding for full display. |
| `p05_tesseract_ocr` | Partially covered | Runtime preflight and whitelist behavior checks added. |
| `p06_yolo_nano` | Partially covered | Detection sorting checks added; model quality still needs dataset-specific validation. |
| `p07_llm_vision` | Partially covered | Provider wiring checks added; end-to-end quality still provider/model dependent. |

### Required Validation Checklist (`p05` / `p06` / `p07`)

- `p05`:
  - verify `tesseract --version` works (or call `/api/pipelines/p05/preflight`)
  - run Quick Test with known labels
  - confirm whitelist behavior on mixed symbols
- `p06`:
  - train or provide a valid `model_path`
  - test varying digit counts and spacing
  - inspect left-to-right detection sorting
- `p07`:
  - set active API key via `/api/keys`
  - verify provider paths used in your environment (`anthropic`/`openai`/`openrouter`/`ollama`)
  - test prompt robustness on glare/blur/partial ROI

## Known Limitations and Practical Notes

- Current automated tests rely heavily on synthetic generated displays.
- Real-world robustness still depends on external datasets and manual error analysis.
- `p06` needs domain training for strong digit performance.
- `p07` incurs network latency and third-party dependency.
- Tesseract quality depends on local language/model packs and preprocessing.
- SQLAlchemy deprecation warnings are present and should be modernized in future cleanup.

## Troubleshooting

- `Could not open requirements file ... requirements.txt`:
  - run commands from `seven-segment-bench/` directory.
- `pytest` not found:
  - use `python -m pytest ...` inside activated venv.
- Tesseract pipeline errors:
  - ensure Tesseract binary is installed and accessible in PATH.
- LLM pipeline key errors:
  - add active key via `/api/keys` using `anthropic`, `openai`, or `openrouter`.
  - for local Ollama, use provider `ollama`; key can be empty, and ensure server is reachable at `http://localhost:11434/v1`.
- OpenRouter connection errors:
  - confirm key is valid and provider is set to `openrouter`.
- YOLO training fails immediately:
  - verify selected dataset has labeled images.
  - ensure `val_ratio` keeps at least one train and one val sample.
  - confirm `base_model` is either `yolov8n.pt` or `yolov8s.pt`.
  - if auto-suggest picked values that are too aggressive for your hardware, reduce `imgsz` and `batch`.
  - verify checkpoint output path is writable under `data/training/p06_yolo_nano`.
- Benchmark says no labeled images:
  - ensure labels exist for images in selected dataset.
- Empty/misaligned predictions:
  - verify ROI placement and `num_digits` assumptions.

## Repo Layout

```text
seven-segment-bench/
  app.py
  config.py
  requirements.txt
  requirements-pi.txt
  setup_pi.sh
  models/
  routes/
  services/
  pipelines/
  templates/
  static/
  tests/
  data/
    uploads/
    thumbnails/
  instance/
    app.db
```

## Suggested Next Experiments

1. Cross-dataset generalization matrix:
   - benchmark each pipeline across multiple dataset sources.
2. Condition-specific evaluation:
   - split by lighting tag, display type, camera distance, and blur level.
3. Hybrid decision policy:
   - confidence-gated fallback chain (for example `p03 -> p02 -> p07`).
4. Expand test suite:
   - add dedicated unit/integration tests for `p05`, `p06`, `p07`.
5. Reproducibility package:
   - pin dataset versions, pipeline configs, and export scripts so repeated runs stay consistent.
