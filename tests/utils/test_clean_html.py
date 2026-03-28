"""Unit tests for html_util cleaning functions."""

from axetract.utils.html_util import clean_html, custom_clean_html

# ===========================================================================
# custom_clean_html
# ===========================================================================


class TestCustomCleanHtml:
    def test_removes_script_tags(self):
        html = "<html><body><script>alert(1)</script><p>Text</p></body></html>"
        result = custom_clean_html(html)
        assert "<script>" not in result
        assert "Text" in result

    def test_removes_style_tags(self):
        html = "<html><head><style>body{color:red}</style></head><body><p>Hi</p></body></html>"
        result = custom_clean_html(html)
        assert "<style>" not in result

    def test_removes_hidden_elements(self):
        html = '<html><body><div style="display:none">Hidden</div><p>Show</p></body></html>'
        result = custom_clean_html(html)
        assert "Hidden" not in result
        assert "Show" in result

    def test_removes_aria_hidden_elements(self):
        html = '<html><body><span aria-hidden="true">Ghost</span><p>Real</p></body></html>'
        result = custom_clean_html(html)
        assert "Ghost" not in result

    def test_removes_onclick_attrs(self):
        html = '<html><body><button onclick="evil()">Click</button></body></html>'
        result = custom_clean_html(html)
        assert "onclick" not in result
        assert "Click" in result

    def test_plain_content_preserved(self):
        html = "<html><body><p>Hello</p></body></html>"
        result = custom_clean_html(html)
        assert "Hello" in result


# ===========================================================================
# clean_html
# ===========================================================================


class TestCleanHtml:
    def test_removes_scripts(self):
        dirty = "<html><head><script>alert(1)</script></head><body><p>Test</p></body></html>"
        cleaned = clean_html(dirty)
        assert "<script>" not in cleaned
        assert "Test" in cleaned

    def test_returns_string(self):
        result = clean_html("<p>Hi</p>")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = clean_html("")
        assert isinstance(result, str)
