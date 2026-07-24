"""Unit tests for the pruner-on/off pairing helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.pruner_delta import (
    aggregate_overhead,
    canonical_key_for_label,
    format_pruner_delta,
    pair_pruner_runs,
    pruner_enabled_for_label,
)


def _metric(batch_size: int, mean_s: float, prune_occ: float) -> Dict[str, Any]:
    """Build a minimal metric dict for pairing tests."""
    return {
        "batch_size": batch_size,
        "micro_batch_size": 4,
        "latency_mean_s": mean_s,
        "docs_per_s": 1.0 / mean_s if mean_s > 0 else 0.0,
        "stage_occupancy_s": {
            "prune": prune_occ,
            "extract": mean_s - prune_occ if mean_s > prune_occ else 0.1,
            "preprocess": 0.1,
            "postprocess": 0.05,
            "setup": 0.0,
        },
    }


def test_pruner_enabled_for_label() -> None:
    """The label parser must return True/False for paired labels and None otherwise."""
    assert pruner_enabled_for_label("vllm/gpu b=1 mb=4 [+pruner]") is True
    assert pruner_enabled_for_label("vllm/gpu b=1 mb=4 [-pruner]") is False
    assert pruner_enabled_for_label("vllm/gpu b=1 mb=4") is None
    # Substring matches must NOT trigger — only the full suffix matters.
    assert pruner_enabled_for_label("[+pruner] foo") is None
    assert pruner_enabled_for_label("foo [-pruner]") is False


def test_canonical_key_for_label() -> None:
    """The canonical key strips the pruner on/off suffix."""
    assert (
        canonical_key_for_label("vllm/gpu b=1 mb=4 [+pruner]")
        == "vllm/gpu b=1 mb=4"
    )
    assert (
        canonical_key_for_label("vllm/gpu b=1 mb=4 [-pruner]")
        == "vllm/gpu b=1 mb=4"
    )
    # No-op for plain labels.
    assert canonical_key_for_label("vllm/gpu b=1 mb=4") == "vllm/gpu b=1 mb=4"


def test_pair_pruner_runs_basic() -> None:
    """Pairing should match on-off twins by canonical key and compute deltas."""
    metrics: List[Dict[str, Any]] = [
        _metric(batch_size=1, mean_s=5.0, prune_occ=3.0),
        _metric(batch_size=1, mean_s=1.0, prune_occ=0.0),
        _metric(batch_size=2, mean_s=8.0, prune_occ=5.0),
        _metric(batch_size=2, mean_s=2.0, prune_occ=0.0),
    ]
    labels = [
        "vllm/gpu b=1 mb=4 [+pruner]",
        "vllm/gpu b=1 mb=4 [-pruner]",
        "vllm/gpu b=2 mb=4 [+pruner]",
        "vllm/gpu b=2 mb=4 [-pruner]",
    ]
    pairs = pair_pruner_runs(metrics, labels)
    assert set(pairs.keys()) == {"vllm/gpu b=1 mb=4", "vllm/gpu b=2 mb=4"}

    p1 = pairs["vllm/gpu b=1 mb=4"]
    assert p1["delta_s"] == 4.0
    assert p1["delta_ratio"] == 5.0
    assert p1["prune_occupancy_s"] == 3.0
    assert p1["on_idx"] == 0
    assert p1["off_idx"] == 1
    assert p1["on_label"] == "vllm/gpu b=1 mb=4 [+pruner]"

    p2 = pairs["vllm/gpu b=2 mb=4"]
    assert p2["delta_s"] == 6.0
    assert p2["delta_ratio"] == 4.0


def test_pair_pruner_runs_single_variant() -> None:
    """A run without pruner-on/off suffix labels should produce no pairs."""
    metrics = [_metric(batch_size=4, mean_s=10.0, prune_occ=6.0)]
    labels = ["vllm/gpu b=4 mb=4"]
    assert pair_pruner_runs(metrics, labels) == {}


def test_pair_pruner_runs_partial_pair_is_dropped() -> None:
    """If only one side of a pair is present, the pair is dropped."""
    metrics = [_metric(batch_size=1, mean_s=5.0, prune_occ=3.0)]
    labels = ["vllm/gpu b=1 mb=4 [+pruner]"]
    assert pair_pruner_runs(metrics, labels) == {}


def test_aggregate_overhead() -> None:
    """Aggregate stats are the simple mean across pairs."""
    metrics = [
        _metric(batch_size=1, mean_s=5.0, prune_occ=3.0),
        _metric(batch_size=1, mean_s=1.0, prune_occ=0.0),
        _metric(batch_size=2, mean_s=8.0, prune_occ=5.0),
        _metric(batch_size=2, mean_s=2.0, prune_occ=0.0),
    ]
    labels = [
        "vllm/gpu b=1 mb=4 [+pruner]",
        "vllm/gpu b=1 mb=4 [-pruner]",
        "vllm/gpu b=2 mb=4 [+pruner]",
        "vllm/gpu b=2 mb=4 [-pruner]",
    ]
    pairs = pair_pruner_runs(metrics, labels)
    agg = aggregate_overhead(pairs)
    assert agg is not None
    assert agg["n_pairs"] == 2
    assert agg["mean_delta_s"] == 5.0
    assert agg["mean_delta_ratio"] == 4.5
    # Prune occupies 3/(3+1+0.1+0.05) ≈ 71.4% and 5/(5+3+0.1+0.05) ≈ 61.2%
    # → mean ≈ 66.3%
    assert 60.0 < agg["mean_prune_pct"] < 75.0


def test_aggregate_overhead_empty() -> None:
    """No pairs → None, not a zero-filled dict."""
    assert aggregate_overhead({}) is None


def test_format_pruner_delta_on_row() -> None:
    """The on-row cell shows +Δs and the ratio."""
    cell = format_pruner_delta(5.0, 1.0, pruner_enabled=True)
    assert cell.startswith("+")
    assert "4.00s" in cell
    assert "×5.0" in cell


def test_format_pruner_delta_off_row() -> None:
    """The off-row cell mirrors the sign so side-by-side scan is natural."""
    cell = format_pruner_delta(5.0, 1.0, pruner_enabled=False)
    assert cell.startswith("−")
    assert "4.00s" in cell


def test_format_pruner_delta_no_twin() -> None:
    """Without a twin (no --pruner=both pair), the cell is the em-dash."""
    assert format_pruner_delta(5.0, 1.0, pruner_enabled=None) == "—"


def test_format_pruner_delta_near_zero() -> None:
    """When the pruner adds <0.01s the cell reads ≈0 (avoids 0.00s noise)."""
    assert format_pruner_delta(1.001, 1.0, pruner_enabled=True) == "≈0"