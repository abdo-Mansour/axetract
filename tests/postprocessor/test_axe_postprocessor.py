"""Unit tests for AXEPostprocessor and its top-level worker functions."""

from unittest.mock import MagicMock, patch

from axetract.data_types import AXESample, Status
from axetract.postprocessor.axe_postprocessor import (
    AXEPostprocessor,
    _recursive_exact_extract_indexed,
    _safe_extract_worker,
)
from axetract.utils.html_util import build_html_search_index

# ===========================================================================
# _recursive_exact_extract_indexed
# ===========================================================================


class TestRecursiveExactExtractIndexed:
    def test_scalar_str_matches_against_index(self):
        html = "<p>matched</p>"
        index = build_html_search_index(html)
        val, xp = _recursive_exact_extract_indexed("matched", index)
        assert val == "matched"
        assert xp is not None

    def test_none_returns_none_none(self):
        index = build_html_search_index("<p>any</p>")
        val, xp = _recursive_exact_extract_indexed(None, index)
        assert val is None
        assert xp is None

    def test_dict_recurses_on_values(self):
        html = "<p>$10</p><span>1kg</span>"
        index = build_html_search_index(html)
        data = {"price": "$10", "weight": "1kg"}
        val, xp = _recursive_exact_extract_indexed(data, index)
        assert isinstance(val, dict)
        assert "price" in val
        assert "weight" in val
        assert isinstance(xp, dict)

    def test_list_recurses_on_items(self):
        html = "<ul><li>item1</li><li>item2</li></ul>"
        index = build_html_search_index(html)
        data = ["item1", "item2"]
        vals, xpaths = _recursive_exact_extract_indexed(data, index)
        assert isinstance(vals, list)
        assert len(vals) == 2
        assert isinstance(xpaths, list)

    def test_empty_index_returns_none(self):
        index = []
        val, xp = _recursive_exact_extract_indexed("something", index)
        assert val is None
        assert xp is None


# ===========================================================================
# _safe_extract_worker
# ===========================================================================


class TestSafeExtractWorker:
    def test_empty_response_returns_empty_string(self):
        result, xpaths = _safe_extract_worker("", "<p>html</p>", "query", False)
        assert result == ""
        assert xpaths is None

    def test_valid_json_parsed(self):
        response = '{"price": "$10", "weight": "1kg"}'
        result, xpaths = _safe_extract_worker(response, "", '{"price":"string"}', False)
        assert isinstance(result, dict)
        assert result.get("price") == "$10"

    def test_markdown_json_parsed(self):
        response = '```json\n{"price": "$10"}\n```'
        result, xpaths = _safe_extract_worker(response, "", '{"price":"string"}', False)
        assert isinstance(result, dict)
        assert result.get("price") == "$10"

    def test_schema_query_does_not_spread(self):
        response = '{"a": "hello", "b": "world"}'
        result, _ = _safe_extract_worker(response, "", '{"a": "string"}', False)
        # Schema → no spread_values, should return dict
        assert isinstance(result, dict)

    def test_qa_query_spreads_values(self):
        response = '{"answer": "yes"}'
        result, _ = _safe_extract_worker(response, "", "What is the answer?", False)
        # QA → spread_values=True → string
        assert isinstance(result, str)

    def test_extract_exact_true_builds_index_and_matches(self):
        response = '{"price": "$10"}'
        html = "<html><body><p>$10</p></body></html>"
        result, xpaths = _safe_extract_worker(response, html, '{"price":"str"}', True)
        assert isinstance(result, dict)
        # Should have matched the $10 text
        assert result.get("price") is not None
        assert xpaths is not None

    def test_extract_exact_true_no_content_returns_error(self):
        response = '{"price": "$10"}'
        result, xpaths = _safe_extract_worker(response, "", '{"price":"str"}', True)
        assert isinstance(result, dict)
        assert "__error__" in result

    def test_none_query_treated_as_non_schema(self):
        response = '{"answer": "yes"}'
        result, _ = _safe_extract_worker(response, "", None, False)
        # None query → not schema → spread_values=True
        assert isinstance(result, str)

    def test_pydantic_query_treated_as_schema(self):
        from pydantic import BaseModel

        class M(BaseModel):
            field: str

        response = '{"field": "value"}'
        result, _ = _safe_extract_worker(response, "", M, False)
        # Pydantic model → is_schema_query=True → spread_values=False → dict
        assert isinstance(result, dict)


# ===========================================================================
# AXEPostprocessor
# ===========================================================================


class TestAXEPostprocessor:
    def _make_sample(
        self, sample_id="1", prediction='{"price": "$10"}', query=None, current_html=""
    ):
        s = AXESample(
            id=sample_id,
            content="<p>content</p>",
            is_content_url=False,
            query=query,
            current_html=current_html,
        )
        s.prediction = prediction
        s.status = Status.SUCCESS
        return s

    def test_empty_samples_returns_empty(self):
        pp = AXEPostprocessor()
        assert pp([]) == []

    def test_stores_config(self):
        pp = AXEPostprocessor(name="my_pp", exact_extraction=False)
        assert pp.name == "my_pp"
        assert pp._exact_extraction is False

    def test_prediction_updated_on_samples(self):
        pp = AXEPostprocessor(exact_extraction=False)
        sample = self._make_sample(
            prediction='{"price": "$10", "weight": "1kg"}',
            query='{"price": "string", "weight": "string"}',
        )
        results = pp([sample])
        # prediction should be updated (string → dict)
        assert results[0].prediction is not None

    def test_multiple_samples_processed(self):
        pp = AXEPostprocessor(exact_extraction=False)
        samples = [
            self._make_sample(
                sample_id=str(i),
                prediction='{"val": "x"}',
                query='{"val": "string"}',
            )
            for i in range(3)
        ]
        results = pp(samples)
        assert len(results) == 3

    def test_non_string_prediction_converted(self):
        """A non-string prediction (e.g., already a dict) should be str-coerced."""
        pp = AXEPostprocessor(exact_extraction=False)
        sample = self._make_sample(query='{"price": "string"}')
        sample.prediction = {"price": "$10"}  # pre-parsed dict
        results = pp([sample])
        assert len(results) == 1

    def test_exact_extraction_uses_thread_pool(self):
        """Verify exact extraction runs without errors using ThreadPoolExecutor."""
        pp = AXEPostprocessor(exact_extraction=True)
        sample = self._make_sample(
            prediction='{"price": "$5"}',
            query='{"price": "string"}',
            current_html="<p>$5</p>",
        )
        results = pp([sample])
        assert len(results) == 1
        # Verify the prediction was processed (should be a dict or have matched values)
        assert results[0].prediction is not None
