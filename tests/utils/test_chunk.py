"""Unit tests for html_util chunking functions."""

from axetract.utils.html_util import chunk_html_content

# ===========================================================================
# chunk_html_content
# ===========================================================================


class TestChunkHtmlContent:
    def test_returns_list(self):
        html = "<div><p>First Paragraph.</p><p>Second Paragraph.</p></div>"
        result = chunk_html_content(html, max_tokens=10)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_html_returns_empty(self):
        result = chunk_html_content("", max_tokens=500)
        assert result == []

    def test_single_small_chunk_not_split(self):
        html = "<p>Short.</p>"
        result = chunk_html_content(html, max_tokens=500)
        # Small enough to fit in one chunk
        assert len(result) >= 1
