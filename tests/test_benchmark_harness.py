"""Unit tests for the benchmark harness helpers."""

from __future__ import annotations

from axetract.data_types import AXESample, Status
from benchmarks.harness import build_batch


def _sample(sample_id: str, content: str = "<html>x</html>") -> AXESample:
    return AXESample(
        id=sample_id,
        content=content,
        is_content_url=False,
        query="q",
    )


def test_build_batch_deep_copies_when_cycling() -> None:
    """Cycling a small corpus must not reuse the same sample objects."""
    corpus = [_sample("a", "A"), _sample("b", "B")]
    batch = build_batch(corpus, batch_size=5)

    assert len(batch) == 5
    assert len({id(s) for s in batch}) == 5
    assert [s.content for s in batch] == ["A", "B", "A", "B", "A"]
    assert batch[0].id != batch[2].id
    assert all(s.status == Status.PENDING for s in batch)
    assert all(s.chunks == [] for s in batch)

    # Mutating one batch item must not affect the corpus or siblings.
    batch[0].content = "MUTATED"
    batch[0].chunks = []  # type: ignore[assignment]
    assert corpus[0].content == "A"
    assert batch[2].content == "A"


def test_build_batch_rejects_empty_corpus() -> None:
    """Empty corpus should raise ValueError."""
    try:
        build_batch([], 1)
        raised = False
    except ValueError:
        raised = True
    assert raised
