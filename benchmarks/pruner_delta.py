"""Pruner on/off comparison helpers.

When the benchmark is run with ``--pruner both``, every ``(batch_size,
micro_batch_size)`` configuration is measured twice — once with the pruner
enabled and once with it skipped. This module pairs those twin runs so the
reports can compute a "pruner overhead" column / KPI / interpretation
bullet.

A *twin* is identified by ``(batch_size, micro_batch_size)``: every
``+pruner`` config shares its key with exactly one ``-pruner`` config (and
vice versa) under the assumption that the sweep iterates the cartesian
product of batch × mb × {pruner on, pruner off}.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Suffixes appended to config labels by ``benchmarks.run`` when --pruner=both.
# Kept here (and not duplicated in run.py) so report code has a single source
# of truth.
PRUNER_ON_SUFFIX = " [+pruner]"
PRUNER_OFF_SUFFIX = " [-pruner]"


def pruner_enabled_for_label(label: str) -> Optional[bool]:
    """Return whether *label* represents a pruner-on / pruner-off run.

    Args:
        label: Config label produced by ``benchmarks.run``.

    Returns:
        ``True`` for ``+pruner`` runs, ``False`` for ``-pruner`` runs,
        ``None`` if the label does not encode a pruner variant (i.e. the
        run was performed with ``--pruner use`` or ``--pruner skip`` only).
    """
    if label.endswith(PRUNER_ON_SUFFIX):
        return True
    if label.endswith(PRUNER_OFF_SUFFIX):
        return False
    return None


def canonical_key_for_label(label: str) -> str:
    """Strip the pruner-on/off suffix from *label* to get the twin key.

    Args:
        label: Config label produced by ``benchmarks.run``.

    Returns:
        The label without any trailing `` [+pruner]`` / `` [-pruner]``
        marker. Safe to call on labels that do not carry the marker.
    """
    if label.endswith(PRUNER_ON_SUFFIX):
        return label[: -len(PRUNER_ON_SUFFIX)]
    if label.endswith(PRUNER_OFF_SUFFIX):
        return label[: -len(PRUNER_OFF_SUFFIX)]
    return label


def pair_pruner_runs(
    metrics_list: List[Dict[str, Any]],
    config_labels: Optional[List[str]] = None,
    raw_list: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Pair ``+pruner`` / ``-pruner`` config twins and compute deltas.

    Args:
        metrics_list: Per-config metric dicts from
            :func:`benchmarks.metrics.compute_run_metrics`. Each entry should
            contain ``batch_size`` and ``micro_batch_size``.
        config_labels: Optional parallel list of human-readable config labels.
            Used both for pairing (via the ``[+pruner]``/``[-pruner]``
            suffix) and for surfacing twin names in the returned dict.
        raw_list: Optional parallel list of raw run records (used for the
            ``pruner_enabled`` flag — labels are the primary source of
            truth, raw is just a cross-check).

    Returns:
        dict: Mapping of ``canonical_key -> {on, off, on_label, off_label,
        delta_s, delta_ratio, prune_occupancy_s}`` for every config that
        has both an on- and off-twin. Configs that only exist on one side
        (or that come from a ``--pruner use`` / ``--pruner skip`` run) are
        omitted.

        ``delta_s`` is the mean latency added by the pruner
        (``mean_on - mean_off``). ``delta_ratio`` is ``mean_on / mean_off``
        (``1.0`` ⇒ free). ``prune_occupancy_s`` is the mean prune-stage
        occupancy from the ``+pruner`` run, useful for the headline
        "pruner accounts for X% of wall time" stat.
    """
    labels = config_labels or [f"run-{i}" for i in range(len(metrics_list))]

    by_key: Dict[str, Dict[str, Any]] = {}

    for i, m in enumerate(metrics_list):
        label = labels[i]
        pruner_on = pruner_enabled_for_label(label)
        if pruner_on is None:
            # Single-variant run — skip; no twin to pair against.
            continue
        key = canonical_key_for_label(label)
        slot = by_key.setdefault(
            key,
            {
                "batch_size": m.get("batch_size"),
                "micro_batch_size": m.get("micro_batch_size"),
                "on": None,
                "off": None,
                "on_label": None,
                "off_label": None,
                "on_idx": None,
                "off_idx": None,
            },
        )
        if pruner_on:
            slot["on"] = m
            slot["on_label"] = label
            slot["on_idx"] = i
        else:
            slot["off"] = m
            slot["off_label"] = label
            slot["off_idx"] = i

    # Drop incomplete pairs and compute deltas.
    complete: Dict[str, Dict[str, Any]] = {}
    for key, slot in by_key.items():
        if slot["on"] is None or slot["off"] is None:
            continue
        on = slot["on"]
        off = slot["off"]
        on_mean = float(on.get("latency_mean_s") or 0.0)
        off_mean = float(off.get("latency_mean_s") or 0.0)
        delta_s = on_mean - off_mean
        delta_ratio = (on_mean / off_mean) if off_mean > 0 else float("inf")
        prune_occ = float((on.get("stage_occupancy_s") or {}).get("prune", 0.0))
        slot.update(
            {
                "delta_s": delta_s,
                "delta_ratio": delta_ratio,
                "prune_occupancy_s": prune_occ,
                "on_mean_s": on_mean,
                "off_mean_s": off_mean,
                "on_docs_per_s": float(on.get("docs_per_s") or 0.0),
                "off_docs_per_s": float(off.get("docs_per_s") or 0.0),
            }
        )
        complete[key] = slot

    return complete


def aggregate_overhead(
    pairs: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Aggregate per-config pruner overheads into headline numbers.

    Args:
        pairs: Output of :func:`pair_pruner_runs`.

    Returns:
        dict with ``mean_delta_ratio``, ``mean_delta_s``, ``mean_prune_pct``
        (average share of wall time spent in the pruner stage across the
        ``+pruner`` runs), and ``n_pairs``. ``None`` when *pairs* is empty.
    """
    if not pairs:
        return None
    ratios = [p["delta_ratio"] for p in pairs.values() if p["delta_ratio"] != float("inf")]
    deltas = [p["delta_s"] for p in pairs.values()]
    prune_pcts: List[float] = []
    for p in pairs.values():
        on = p["on"]
        wall = float(on.get("latency_mean_s") or 0.0)
        if wall > 0:
            prune_pcts.append(p["prune_occupancy_s"] / wall * 100.0)
    if not ratios:
        return None
    return {
        "mean_delta_ratio": sum(ratios) / len(ratios),
        "mean_delta_s": sum(deltas) / len(deltas),
        "mean_prune_pct": (sum(prune_pcts) / len(prune_pcts)) if prune_pcts else 0.0,
        "n_pairs": len(pairs),
    }


def format_pruner_delta(
    on_mean_s: Optional[float],
    off_mean_s: Optional[float],
    *,
    pruner_enabled: Optional[bool] = None,
    precision_time: int = 2,
    precision_ratio: int = 1,
) -> str:
    """Format a per-row "Pruner Δ" cell.

    The function is designed to be called from report loops that walk
    ``metrics_list`` / ``config_labels`` in order. For each row it looks up
    the matching twin (configurations that differ only in their
    ``[+pruner]`` / ``[-pruner]`` suffix) and computes the overhead.

    Args:
        on_mean_s: Mean latency of the ``+pruner`` twin (or the current
            row's latency if *pruner_enabled* is ``True``).
        off_mean_s: Mean latency of the ``-pruner`` twin (or the current
            row's latency if *pruner_enabled* is ``False``). Pass ``None``
            if no twin exists (e.g. the run was performed with
            ``--pruner use`` or ``--pruner skip`` only).
        pruner_enabled: Whether the *current* row is a ``+pruner`` run.
            When ``False``, the row's own latency is in *off_mean_s* and
            the function returns a mirror-formatted string (e.g.
            ``"−4.00s (×0.2)"``) so a side-by-side scan of the two twin
            rows shows the same absolute delta with opposite signs.
            When ``None`` (no twin exists), the function returns ``"—"``.
        precision_time: Decimal places for the seconds component.
        precision_ratio: Decimal places for the ×N multiplier.

    Returns:
        A compact string like ``"+12.34s (×4.2)"`` for a ``+pruner`` row
        with a twin, ``"−12.34s (×0.2)"`` for the matching ``−pruner``
        row, or ``"—"`` when no twin exists.
    """
    if pruner_enabled is None:
        return "—"
    if on_mean_s is None or off_mean_s is None or off_mean_s <= 0:
        return "—"
    delta = on_mean_s - off_mean_s
    ratio = on_mean_s / off_mean_s
    if abs(delta) < 10 ** (-precision_time) and abs(ratio - 1.0) < 10 ** (-precision_ratio):
        return "≈0"
    sign = "+" if delta >= 0 else "−"
    # For the off-row, mirror the sign so the two rows show the same
    # magnitude with opposite symbols — easy visual scan.
    if not pruner_enabled:
        sign = "+" if delta <= 0 else "−"
    return f"{sign}{abs(delta):.{precision_time}f}s (×{ratio:.{precision_ratio}f})"


# Re-export the Optional type so other modules can import cleanly if needed.
__all__ = [
    "PRUNER_ON_SUFFIX",
    "PRUNER_OFF_SUFFIX",
    "pruner_enabled_for_label",
    "canonical_key_for_label",
    "pair_pruner_runs",
    "aggregate_overhead",
    "format_pruner_delta",
]