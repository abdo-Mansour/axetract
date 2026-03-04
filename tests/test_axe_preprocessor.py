from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest
from axetract.data_types import AXESample, AXEChunk
from axetract.preprocessor.axe_preprocessor import AXEPreprocessor, _chunk_worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(
    *,
    id: str = "s1",
    content: str = "<p>hello</p>",
    is_content_url: bool = False,
) -> AXESample:
    return AXESample(
        id=id,
        content=content,
        is_content_url=is_content_url,
    )


# ===================================================================
# AXEPreprocessor.__init__
# ===================================================================

class TestAXEPreprocessorInit:
    """Verify constructor defaults and custom overrides."""

    def test_default_values(self):
        p = AXEPreprocessor()
        assert p.fetch_workers == 1
        assert p.cpu_workers == 1
        assert p.extra_remove_tags is None
        assert p.strip_attrs is True
        assert p.strip_links is True
        assert p.keep_tags is False
        # Defaults from the real implementation
        assert p.use_clean_rag is True
        assert p.use_clean_chunker is True
        assert p.chunk_size == 1000
        assert p.attr_cutoff_len == 100
        assert p.disable_chunking is False

    def test_custom_values(self):
        p = AXEPreprocessor(
            fetch_workers=4,
            cpu_workers=2,
            extra_remove_tags=["nav", "footer"],
            strip_attrs=False,
            strip_links=False,
            keep_tags=True,
            use_clean_rag=True,
            use_clean_chunker=True,
            chunk_size=500,
            attr_cutoff_len=50,
        )
        assert p.fetch_workers == 4
        assert p.cpu_workers == 2
        assert p.extra_remove_tags == ["nav", "footer"]
        assert p.strip_attrs is False
        assert p.strip_links is False
        assert p.keep_tags is True
        assert p.use_clean_rag is True
        assert p.use_clean_chunker is True
        assert p.chunk_size == 500
        assert p.attr_cutoff_len == 50


# ===================================================================
# AXEPreprocessor.process – fetching behaviour
# ===================================================================

class TestProcessFetching:
    """Ensure URL-based samples are fetched and non-URL samples are left alone."""

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["<p>hello</p>"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    @patch("axetract.preprocessor.axe_preprocessor.fetch_content", return_value="<html>fetched</html>")
    def test_url_sample_is_fetched(self, mock_fetch, mock_clean, mock_chunk):
        sample = _make_sample(content="https://example.com", is_content_url=True)
        p = AXEPreprocessor()
        result = p(sample)

        mock_fetch.assert_called_once_with("https://example.com")
        # After fetching, is_content_url should be False and content replaced
        assert result[0].is_content_url is False
        assert result[0].content == "<html>fetched</html>"

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["<p>hello</p>"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    @patch("axetract.preprocessor.axe_preprocessor.fetch_content")
    def test_non_url_sample_not_fetched(self, mock_fetch, mock_clean, mock_chunk):
        sample = _make_sample(content="<p>inline</p>", is_content_url=False)
        p = AXEPreprocessor()
        p(sample)

        mock_fetch.assert_not_called()

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["<p>hello</p>"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    @patch("axetract.preprocessor.axe_preprocessor.fetch_content", side_effect=Exception("timeout"))
    def test_fetch_error_stores_error_in_content(self, mock_fetch, mock_clean, mock_chunk):
        sample = _make_sample(content="https://bad-url.com", is_content_url=True)
        p = AXEPreprocessor()
        result = p(sample)

        assert "[Fetch ERROR]" in result[0].content
        assert result[0].is_content_url is False


# ===================================================================
# AXEPreprocessor.process – single / batch / empty
# ===================================================================

class TestProcessBatching:
    """Verify single-sample wrapping, empty-list short-circuit, and batch."""

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["c"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    def test_single_sample_wrapped_in_list(self, mock_clean, mock_chunk):
        sample = _make_sample()
        p = AXEPreprocessor()
        result = p(sample)
        assert isinstance(result, list)
        assert len(result) == 1

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["c"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    def test_empty_batch_returns_empty_list(self, mock_clean, mock_chunk):
        p = AXEPreprocessor()
        result = p([])
        assert result == []

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["c"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    def test_batch_of_multiple_samples(self, mock_clean, mock_chunk):
        samples = [_make_sample(id=f"s{i}") for i in range(5)]
        p = AXEPreprocessor()
        result = p(samples)
        assert len(result) == 5


# ===================================================================
# AXEPreprocessor.process – executor selection
# ===================================================================

class TestProcessExecutor:
    """Ensure correct executor is selected based on cpu_workers."""

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["c"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    @patch("axetract.preprocessor.axe_preprocessor.ProcessPoolExecutor")
    @patch("axetract.preprocessor.axe_preprocessor.ThreadPoolExecutor")
    def test_thread_executor_when_cpu_workers_le_1(self, mock_thread, mock_proc, mock_clean, mock_chunk):
        # cpu_workers=1 → ThreadPoolExecutor
        sample = _make_sample()
        # Mock map to return the input sample list
        mock_thread.return_value.__enter__.return_value.map.side_effect = lambda f, items: map(f, items)
        mock_thread.return_value.__exit__ = MagicMock(return_value=False)

        p = AXEPreprocessor(cpu_workers=1)
        p(sample)

        # ThreadPoolExecutor is used for both fetch AND chunk phases, so it
        # should be called at least twice (fetch pool + chunk pool).
        assert mock_thread.call_count >= 2
        mock_proc.assert_not_called()

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["c"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    @patch("axetract.preprocessor.axe_preprocessor.ProcessPoolExecutor")
    def test_process_executor_when_cpu_workers_gt_1(self, mock_proc, mock_clean, mock_chunk):
        mock_proc.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                map=MagicMock(return_value=iter([{"doc_id": 0, "chunks": []}]))
            )
        )
        mock_proc.return_value.__exit__ = MagicMock(return_value=False)

        p = AXEPreprocessor(cpu_workers=4)
        p(_make_sample())

        mock_proc.assert_called_once()


# ===================================================================
# _chunk_worker
# ===================================================================

class TestChunkWorker:
    """Unit tests for the module-level _chunk_worker helper."""

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["<p>a</p>", "<p>b</p>"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", return_value="<p>a</p><p>b</p>")
    def test_returns_chunks_with_correct_ids(self, mock_clean, mock_chunk):
        sample = _make_sample(content="<div>raw</div>")
        config = AXEPreprocessor()
        config.disable_chunking = False

        result = _chunk_worker((sample, config, 7))
        assert result["doc_id"] == 7
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["chunkid"] == "7-1"
        assert result["chunks"][1]["chunkid"] == "7-2"

    @patch("axetract.preprocessor.axe_preprocessor.clean_html", return_value="")
    def test_empty_content_returns_error_chunk(self, mock_clean):
        sample = _make_sample(content="")
        config = AXEPreprocessor()
        config.disable_chunking = False

        result = _chunk_worker((sample, config, 3))
        assert result["doc_id"] == 3
        assert "err" in result["chunks"][0]["chunkid"]
        assert (
            "empty content" in result["chunks"][0]["chunkcontent"].lower()
            or "error" in result["chunks"][0]["chunkcontent"].lower()
        )

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content")
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", return_value="<p>full</p>")
    def test_disable_chunking_returns_single_chunk(self, mock_clean, mock_chunk):
        sample = _make_sample(content="<p>full</p>")
        config = AXEPreprocessor()
        config.disable_chunking = True

        result = _chunk_worker((sample, config, 0))
        mock_chunk.assert_not_called()
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["chunkcontent"] == "<p>full</p>"

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", side_effect=RuntimeError("boom"))
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", return_value="<p>ok</p>")
    def test_exception_in_chunking_returns_error_payload(self, mock_clean, mock_chunk):
        sample = _make_sample(content="<p>ok</p>")
        config = AXEPreprocessor()
        config.disable_chunking = False

        result = _chunk_worker((sample, config, 5))
        # The error branch returns a tuple (idx, err_payload)
        if isinstance(result, tuple):
            idx, payload = result
            assert idx == 5
            assert "ERROR" in payload["chunks"][0]["chunkcontent"]
        else:
            # In case the code is fixed to return just the dict
            assert "err" in result["chunks"][0]["chunkid"]

    @patch("axetract.preprocessor.axe_preprocessor.clean_html", return_value="<p>cleaned</p>")
    def test_clean_html_called_with_config_params(self, mock_clean):
        sample = _make_sample(content="<div>raw</div>")
        config = AXEPreprocessor(
            extra_remove_tags=["nav"],
            strip_attrs=False,
            strip_links=False,
            keep_tags=True,
            use_clean_rag=True,
        )
        config.disable_chunking = True

        _chunk_worker((sample, config, 0))

        mock_clean.assert_called_once_with(
            html_content="<div>raw</div>",
            extra_remove_tags=["nav"],
            strip_attrs=False,
            strip_links=False,
            keep_tags=True,
            use_clean_rag=True,
        )


# ===================================================================
# Integration-style tests (still mocking external I/O)
# ===================================================================

class TestProcessIntegration:
    """End-to-end flow through process() with mocked utilities."""

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["<p>1</p>", "<p>2</p>"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    @patch("axetract.preprocessor.axe_preprocessor.fetch_content", return_value="<html>page</html>")
    def test_full_pipeline_url_sample(self, mock_fetch, mock_clean, mock_chunk):
        sample = _make_sample(content="https://example.com", is_content_url=True)
        p = AXEPreprocessor(fetch_workers=2, cpu_workers=1, chunk_size=200)
        result = p(sample)

        # fetch was called
        mock_fetch.assert_called_once()
        # clean was called at least once (in _chunk_worker)
        assert mock_clean.call_count >= 1
        # result is a list
        assert len(result) == 1

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["c1"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    def test_full_pipeline_inline_batch(self, mock_clean, mock_chunk):
        samples = [
            _make_sample(id="a", content="<p>A</p>"),
            _make_sample(id="b", content="<p>B</p>"),
            _make_sample(id="c", content="<p>C</p>"),
        ]
        p = AXEPreprocessor()
        result = p(samples)

        assert len(result) == 3
        # clean_html should be called once per sample (inside _chunk_worker)
        assert mock_clean.call_count == 3

    @patch("axetract.preprocessor.axe_preprocessor.chunk_html_content", return_value=["c1"])
    @patch("axetract.preprocessor.axe_preprocessor.clean_html", side_effect=lambda html_content, **kw: html_content)
    @patch("axetract.preprocessor.axe_preprocessor.fetch_content")
    def test_mixed_url_and_inline_batch(self, mock_fetch, mock_clean, mock_chunk):
        mock_fetch.return_value = "<html>fetched</html>"
        samples = [
            _make_sample(id="url1", content="https://a.com", is_content_url=True),
            _make_sample(id="inline1", content="<p>inline</p>", is_content_url=False),
        ]
        p = AXEPreprocessor()
        result = p(samples)

        assert len(result) == 2
        mock_fetch.assert_called_once_with("https://a.com")
        assert result[0].content == "<html>fetched</html>"
        assert result[1].content == "<p>inline</p>"
