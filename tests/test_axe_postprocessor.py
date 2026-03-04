"""Unit tests for AXEPostprocessor and its top-level worker functions."""
import pytest
from unittest.mock import MagicMock, patch

from axetract.postprocessor.axe_postprocessor import (
    AXEPostprocessor,
    _recursive_exact_extract,
    _safe_extract_worker,
)
from axetract.data_types import AXESample, AXEChunk, Status


# ===========================================================================
# _recursive_exact_extract
# ===========================================================================

class TestRecursiveExactExtract:

    @patch("axetract.postprocessor.axe_postprocessor.find_closest_html_node")
    def test_scalar_str_calls_find_closest(self, mock_find):
        mock_find.return_value = {"text": "matched", "xpath": "/html/body/p"}
        val, xp = _recursive_exact_extract("hello", "<p>hello</p>")
        assert val == "matched"
        assert xp == "/html/body/p"

    def test_none_returns_none_none(self):
        val, xp = _recursive_exact_extract(None, "<p>any</p>")
        assert val is None
        assert xp is None

    @patch("axetract.postprocessor.axe_postprocessor.find_closest_html_node")
    def test_dict_recurses_on_values(self, mock_find):
        mock_find.return_value = {"text": "found", "xpath": "/p"}
        data = {"price": "$10", "weight": "1kg"}
        val, xp = _recursive_exact_extract(data, "<p>$10</p><span>1kg</span>")
        assert isinstance(val, dict)
        assert "price" in val
        assert "weight" in val
        assert isinstance(xp, dict)

    @patch("axetract.postprocessor.axe_postprocessor.find_closest_html_node")
    def test_list_recurses_on_items(self, mock_find):
        mock_find.return_value = {"text": "item", "xpath": "/li"}
        data = ["item1", "item2"]
        vals, xpaths = _recursive_exact_extract(data, "<ul><li>item1</li><li>item2</li></ul>")
        assert isinstance(vals, list)
        assert len(vals) == 2
        assert isinstance(xpaths, list)

    @patch("axetract.postprocessor.axe_postprocessor.find_closest_html_node")
    def test_find_closest_none_result(self, mock_find):
        # When find_closest_html_node returns None
        mock_find.return_value = None
        val, xp = _recursive_exact_extract("something", "<p>content</p>")
        assert val is None
        assert xp is None

    @patch("axetract.postprocessor.axe_postprocessor.find_closest_html_node")
    def test_find_closest_missing_fields(self, mock_find):
        # Returns dict without 'text' or 'xpath'
        mock_find.return_value = {}
        val, xp = _recursive_exact_extract("something", "<p>content</p>")
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

    def test_extract_exact_true_calls_recursive(self):
        response = '{"price": "$10"}'
        html = "<html><body><p>$10</p></body></html>"
        with patch("axetract.postprocessor.axe_postprocessor._recursive_exact_extract") as mock_rec:
            mock_rec.return_value = ({"price": "$10"}, {"price": "/html/body/p"})
            result, xpaths = _safe_extract_worker(response, html, '{"price":"str"}', True)
            mock_rec.assert_called_once()
            assert xpaths == {"price": "/html/body/p"}

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

    def _make_sample(self, sample_id="1", prediction='{"price": "$10"}', query=None, current_html=""):
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

    @patch("axetract.postprocessor.axe_postprocessor.ProcessPoolExecutor")
    def test_uses_process_pool_for_exact_extraction(self, mock_exec):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.map.return_value = iter([('{"price":"$5"}', None)])
        mock_exec.return_value = mock_ctx

        pp = AXEPostprocessor(exact_extraction=True)
        sample = self._make_sample(
            prediction='{"price": "$5"}',
            query='{"price": "string"}',
            current_html="<p>$5</p>",
        )
        # Just ensure it runs without raising
        with patch("axetract.postprocessor.axe_postprocessor._safe_extract_worker", return_value=({"price": "$5"}, None)):
            results = pp([sample])
        assert len(results) == 1
