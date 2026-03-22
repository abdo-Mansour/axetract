"""Unit tests for html_util normalization functions."""

from axetract.utils.html_util import normalize_html_text


# ===========================================================================
# normalize_html_text
# ===========================================================================


class TestNormalizeHtmlText:
    def test_empty_string(self):
        assert normalize_html_text("") == ""

    def test_none_returns_empty(self):
        assert normalize_html_text(None) == ""

    def test_collapses_multiple_spaces(self):
        assert normalize_html_text("a  b   c") == "a b c"

    def test_strips_leading_trailing(self):
        assert normalize_html_text("  hello  ") == "hello"

    def test_unicode_whitespace_replaced(self):
        # non-breaking space \u00a0
        result = normalize_html_text("price\u00a0$10")
        assert " " in result
        assert "\u00a0" not in result

    def test_normal_text_unchanged(self):
        assert normalize_html_text("Hello World") == "Hello World"
