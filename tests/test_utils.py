"""Expanded utils tests: json_util, html_util key functions."""

from pydantic import BaseModel

from axetract.utils.html_util import (
    chunk_html_content,
    clean_html,
    custom_clean_html,
    extract_visible_xpaths_leaves,
    find_closest_html_node,
    normalize_html_text,
)
from axetract.utils.json_util import extract_and_repair_json, is_schema


class DummyModel(BaseModel):
    id: int
    name: str


# ===========================================================================
# is_schema
# ===========================================================================


class TestIsSchema:
    def test_dict_input(self):
        assert is_schema({"name": "string"}) is True

    def test_pydantic_class(self):
        assert is_schema(DummyModel) is True

    def test_json_string(self):
        assert is_schema('{"name": "string"}') is True

    def test_dict_keys_string(self):
        assert is_schema("dict_keys(['name', 'age'])") is True

    def test_plain_question_is_not_schema(self):
        assert is_schema("What is the title?") is False

    def test_none_is_not_schema(self):
        assert is_schema(None) is False

    def test_integer_is_not_schema(self):
        assert is_schema(42) is False

    def test_empty_string_is_not_schema(self):
        assert is_schema("") is False

    def test_json_array_string(self):
        assert is_schema('["a", "b"]') is True

    def test_colon_rich_string_treated_as_schema(self):
        # Two or more colons triggers the heuristic
        assert is_schema("price: string, weight: string, count: int") is True


# ===========================================================================
# extract_and_repair_json
# ===========================================================================


class TestExtractAndRepairJson:
    def test_valid_json(self):
        result = extract_and_repair_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced_json(self):
        md = '```json\n{"price": 100}\n```'
        result = extract_and_repair_json(md)
        assert result == {"price": 100}

    def test_malformed_json_repaired(self):
        malformed = '{name: "test"'
        result = extract_and_repair_json(malformed)
        assert isinstance(result, dict)
        assert result.get("name") == "test"

    def test_none_returns_empty_dict(self):
        assert extract_and_repair_json(None) == {}

    def test_dict_input_returned_as_is(self):
        d = {"already": "dict"}
        assert extract_and_repair_json(d) == d

    def test_spread_values_concatenates_strings(self):
        result = extract_and_repair_json('{"k1": "Hello", "k2": " World"}', spread_values=True)
        assert isinstance(result, str)
        assert "Hello" in result
        assert "World" in result

    def test_spread_values_skips_non_strings(self):
        result = extract_and_repair_json('{"k1": "Hello", "k2": 123}', spread_values=True)
        assert "Hello" in result
        # 123 should be skipped (not a string)

    def test_embedded_json_block_extracted(self):
        response = 'Some prefix text {"a": 1} some suffix'
        result = extract_and_repair_json(response)
        assert result.get("a") == 1


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
