"""Benchmark metric computation."""

import numpy as np
from collections import defaultdict
from models.benchmark import BenchmarkRun, BenchmarkResult


def compute_run_metrics(run_id: int) -> dict:
    """Compute all metrics for a benchmark run."""
    run = BenchmarkRun.query.get(run_id)
    if not run:
        return {}

    results = BenchmarkResult.query.filter_by(run_id=run_id).all()
    if not results:
        return {'pipelines': {}}

    # Group results by pipeline
    by_pipeline = defaultdict(list)
    for r in results:
        by_pipeline[r.pipeline_slug].append(r)

    pipeline_metrics = {}
    for slug, pipe_results in by_pipeline.items():
        total = len(pipe_results)
        errors = sum(1 for r in pipe_results if r.error_message)
        valid = [r for r in pipe_results if not r.error_message]

        correct = sum(1 for r in valid if r.is_correct)
        char_correct_total = sum(r.char_correct for r in valid)
        char_total = sum(r.char_total for r in valid)

        latencies = [r.latency_ms for r in valid]
        latencies_sorted = sorted(latencies) if latencies else [0]

        # Confusion matrix (10x10 for digits 0-9)
        confusion = np.zeros((10, 10), dtype=int)
        for r in valid:
            gt = r.ground_truth
            pred = r.predicted
            for i in range(min(len(gt), len(pred))):
                if gt[i].isdigit() and pred[i].isdigit():
                    confusion[int(gt[i])][int(pred[i])] += 1

        pipeline_metrics[slug] = {
            'total': total,
            'errors': errors,
            'valid': len(valid),
            'string_accuracy': round(correct / len(valid) * 100, 2) if valid else 0,
            'char_accuracy': round(char_correct_total / char_total * 100, 2) if char_total else 0,
            'correct': correct,
            'char_correct': char_correct_total,
            'char_total': char_total,
            'latency_mean': round(np.mean(latencies), 2) if latencies else 0,
            'latency_median': round(np.median(latencies), 2) if latencies else 0,
            'latency_p95': round(latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0, 2),
            'confusion_matrix': confusion.tolist(),
        }

    return {
        'run_id': run_id,
        'run_name': run.name,
        'pipelines': pipeline_metrics,
        'pipeline_slugs': list(by_pipeline.keys()),
    }


def compare_char_accuracy(predicted: str, ground_truth: str) -> tuple[int, int]:
    """Compare predicted vs ground truth character-by-character.

    Returns (chars_correct, total_chars).
    """
    total = max(len(predicted), len(ground_truth))
    if total == 0:
        return 0, 0
    correct = 0
    for i in range(min(len(predicted), len(ground_truth))):
        if predicted[i] == ground_truth[i]:
            correct += 1
    return correct, total
