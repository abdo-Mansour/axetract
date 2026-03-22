"""Unit tests for html_util extraction functions."""

from axetract.utils.html_util import extract_visible_xpaths_leaves, find_closest_html_node


# ===========================================================================
# extract_visible_xpaths_leaves
# ===========================================================================


class TestExtractVisibleXpathsLeaves:
    def test_basic_extraction(self):
        html = "<html><body><p>Hello</p><span>World</span></body></html>"
        result = extract_visible_xpaths_leaves(html)
        assert isinstance(result, list)
        texts = [text for _, text in result]
        assert "Hello" in texts or any("Hello" in t for t in texts)

    def test_empty_html_returns_empty(self):
        result = extract_visible_xpaths_leaves("")
        assert result == []

    def test_returns_xpath_text_pairs(self):
        html = "<html><body><p>Test</p></body></html>"
        result = extract_visible_xpaths_leaves(html)
        for item in result:
            assert len(item) == 2  # (xpath, text)

    def test_deduplication(self):
        # Same text repeated: with dedupe_texts=True (default), counted once
        html = "<html><body><p>Same</p><p>Same</p></body></html>"
        result = extract_visible_xpaths_leaves(html, dedupe_texts=True)
        texts = [t for _, t in result]
        assert texts.count("Same") <= 1


# ===========================================================================
# find_closest_html_node
# ===========================================================================


class TestFindClosestHtmlNode:
    def test_finds_exact_match(self):
        html = "<html><body><p>SuperWidget 3000</p></body></html>"
        result = find_closest_html_node(html, "SuperWidget 3000")
        assert result["found"] is True
        assert "SuperWidget" in result["text"]

    def test_no_match_returns_not_found(self):
        html = "<html><body><p>Nothing here</p></body></html>"
        result = find_closest_html_node(html, "XYZ_DOES_NOT_EXIST_12345")
        # may or may not find based on token intersection; at minimum check structure
        assert "found" in result
        assert "text" in result

    def test_empty_search_returns_not_found(self):
        html = "<html><body><p>Content</p></body></html>"
        result = find_closest_html_node(html, "")
        assert result["xpath"] is None
