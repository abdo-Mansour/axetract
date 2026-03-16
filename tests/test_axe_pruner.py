"""Unit tests for axe_pruner module."""

from unittest.mock import MagicMock, patch

from axetract.data_types import AXEChunk, AXESample
from axetract.prompts.pruner_prompt import PRUNER_PROMPT
from axetract.pruner.axe_pruner import (
    AXEPruner,
    _escape_single_quotes,
    _longest_common_xpath_prefix,
    _remove_prefix_from_xpath,
    _worker_filter_prep,
    _worker_merge_html,
    generate_pruner_prompt,
)

# ===========================================================================
# _longest_common_xpath_prefix
# ===========================================================================


class TestLongestCommonXpathPrefix:
    def test_common_prefix(self):
        xpaths = [
            "/html/body/div[1]/p",
            "/html/body/div[1]/span",
            "/html/body/div[1]/img",
        ]
        assert _longest_common_xpath_prefix(xpaths) == "/html/body/div[1]"

    def test_no_common_prefix_returns_root(self):
        xpaths = ["/a/b", "/c/d"]
        prefix = _longest_common_xpath_prefix(xpaths)
        assert prefix == "/"

    def test_single_xpath(self):
        prefix = _longest_common_xpath_prefix(["/html/body/p"])
        assert prefix == "/html/body/p"

    def test_empty_list(self):
        assert _longest_common_xpath_prefix([]) == "/"

    def test_identical_paths(self):
        xpaths = ["/html/body/div", "/html/body/div"]
        assert _longest_common_xpath_prefix(xpaths) == "/html/body/div"

    def test_xpath_without_leading_slash(self):
        # Should normalise missing leading slash
        prefix = _longest_common_xpath_prefix(["html/body/p", "html/body/span"])
        assert prefix.startswith("/html/body")


# ===========================================================================
# _remove_prefix_from_xpath
# ===========================================================================


class TestRemovePrefixFromXpath:
    def test_basic_removal(self):
        assert _remove_prefix_from_xpath("/html/body/div[1]/p", "/html/body/div[1]") == "/p"

    def test_xpath_equals_prefix(self):
        assert _remove_prefix_from_xpath("/html/body", "/html/body") == "/"

    def test_root_prefix_unchanged(self):
        assert _remove_prefix_from_xpath("/html/body/p", "/") == "/html/body/p"

    def test_none_xpath_returns_root(self):
        assert _remove_prefix_from_xpath(None, "/html") == "/"

    def test_empty_xpath_returns_root(self):
        assert _remove_prefix_from_xpath("", "/html") == "/"

    def test_no_matching_prefix(self):
        # xpath doesn't start with prefix → returned as-is
        result = _remove_prefix_from_xpath("/a/b/c", "/x/y")
        assert result == "/a/b/c"


# ===========================================================================
# _escape_single_quotes
# ===========================================================================


class TestEscapeSingleQuotes:
    def test_none_returns_empty_string(self):
        assert _escape_single_quotes(None) == ""

    def test_empty_string(self):
        assert _escape_single_quotes("") == ""

    def test_no_quotes(self):
        assert _escape_single_quotes("hello world") == "hello world"

    def test_single_quote_escaped(self):
        assert _escape_single_quotes("it's") == "it\\'s"

    def test_multiple_quotes(self):
        result = _escape_single_quotes("don't can't")
        assert result == "don\\'t can\\'t"


# ===========================================================================
# generate_pruner_prompt
# ===========================================================================


class TestGeneratePrunerPrompt:
    TEMPLATE = "Query: {query}. Content: {content}"

    def test_basic_tuple_pairs(self):
        pairs = [("/html/body/p", "Hello")]
        prompt = generate_pruner_prompt(pairs, "Find greeting", self.TEMPLATE)
        assert "Query: Find greeting." in prompt
        assert "Hello" in prompt

    def test_dict_pairs(self):
        pairs = [{"xpath": "/html/p", "content": "Price: $10"}]
        prompt = generate_pruner_prompt(pairs, "price?", self.TEMPLATE)
        assert "Price: $10" in prompt

    def test_none_pair_handled(self):
        pairs = [None, ("/html/body/p", "text")]
        # Should not raise; None pairs get normalised to ("", "")
        prompt = generate_pruner_prompt(pairs, "q", self.TEMPLATE)
        assert isinstance(prompt, str)

    def test_relative_xpath_in_output(self):
        pairs = [
            ("/html/body/div/p", "First"),
            ("/html/body/div/span", "Second"),
        ]
        prompt = generate_pruner_prompt(pairs, "q", self.TEMPLATE)
        # Common prefix is /html/body/div so relative paths appear
        assert "/p" in prompt or "/span" in prompt

    def test_empty_pairs_list(self):
        prompt = generate_pruner_prompt([], "q", self.TEMPLATE)
        assert isinstance(prompt, str)


# ===========================================================================
# _worker_filter_prep  (top-level worker — no multiprocessing needed)
# ===========================================================================


class TestWorkerFilterPrep:
    @patch("axetract.pruner.axe_pruner.SmartHTMLProcessor")
    def test_returns_tuple_of_chunks_and_prompt(self, mock_processor_cls):
        mock_processor = MagicMock()
        mock_processor.extract_chunks.return_value = [{"xpath": "/html/body/p", "content": "Hello"}]
        mock_processor_cls.return_value = mock_processor

        chunk_content = "<p>Hello</p>"
        query = "What is said?"
        template = "Query: {query}. Content: {content}"

        chunk_xpaths, prompt = _worker_filter_prep((chunk_content, query, template))

        assert isinstance(chunk_xpaths, list)
        assert isinstance(prompt, str)
        assert "What is said?" in prompt

    @patch("axetract.pruner.axe_pruner.SmartHTMLProcessor")
    def test_empty_html_returns_empty_chunks(self, mock_processor_cls):
        mock_processor = MagicMock()
        mock_processor.extract_chunks.return_value = []
        mock_processor_cls.return_value = mock_processor

        chunk_xpaths, prompt = _worker_filter_prep(("", "q", "Q:{query} C:{content}"))
        assert chunk_xpaths == []


# ===========================================================================
# _worker_merge_html  (top-level worker)
# ===========================================================================


class TestWorkerMergeHtml:
    @patch("axetract.utils.html_util.merge_html_chunks", return_value="<html>Merged</html>")
    def test_newlines_stripped_from_result(self, mock_merge):
        result = _worker_merge_html(([[]], "<p>fallback</p>"))
        assert "\n" not in result

    @patch(
        "axetract.pruner.axe_pruner.merge_html_chunks", return_value="<html>\nLine1\nLine2\n</html>"
    )
    def test_merge_called_with_correct_args(self, mock_merge):
        chunks = [[{"xpath": "/p", "content": "x"}]]
        content = "<p>raw</p>"
        _worker_merge_html((chunks, content))
        mock_merge.assert_called_once_with(chunks, content)


# ===========================================================================
# AXEPruner integration (mocked multiprocessing)
# ===========================================================================


class TestAXEPruner:
    def _make_sample(self, sample_id="0"):
        return AXESample(
            id=sample_id,
            content="<html><body><p>Test</p></body></html>",
            query="What is it?",
            is_content_url=False,
            chunks=[AXEChunk(chunkid=f"{sample_id}-1", content="<p>Test</p>")],
        )

    def test_empty_batch_returns_empty(self):
        mock_llm = MagicMock()
        pruner = AXEPruner(llm_pruner_client=mock_llm, llm_pruner_prompt=PRUNER_PROMPT)
        result = pruner([])
        assert result == []
        mock_llm.call_batch.assert_not_called()

    @patch("axetract.pruner.axe_pruner.ThreadPoolExecutor")
    def test_filter_updates_current_html(self, mock_executor):
        mock_llm = MagicMock()
        mock_llm.call_batch.return_value = ["[0]"]

        mock_ctx = MagicMock()
        mock_ctx.map.side_effect = [
            [([{"xpath": "/p", "content": "Test"}], "Mock Prompt")],
            ["<html>Merged</html>"],
        ]
        mock_executor.return_value.__enter__.return_value = mock_ctx

        pruner = AXEPruner(llm_pruner_client=mock_llm, llm_pruner_prompt=PRUNER_PROMPT)
        sample = self._make_sample()
        results = pruner([sample])

        assert len(results) == 1
        assert results[0].current_html == "<html>Merged</html>"
        mock_llm.call_batch.assert_called_once()

    @patch("axetract.pruner.axe_pruner.ThreadPoolExecutor")
    def test_empty_llm_response_keeps_all_chunks(self, mock_executor):
        """When LLM returns empty string, all xpaths are preserved."""
        mock_llm = MagicMock()
        mock_llm.call_batch.return_value = [""]  # empty → keep all

        mock_ctx = MagicMock()
        mock_ctx.map.side_effect = [
            [([{"xpath": "/p", "content": "Test"}], "Prompt")],
            ["<html>Full</html>"],
        ]
        mock_executor.return_value.__enter__.return_value = mock_ctx

        pruner = AXEPruner(llm_pruner_client=mock_llm, llm_pruner_prompt=PRUNER_PROMPT)
        results = pruner([self._make_sample()])
        assert len(results) == 1
        assert results[0].current_html == "<html>Full</html>"

    def test_pruner_stores_config(self):
        mock_llm = MagicMock()
        pruner = AXEPruner(
            llm_pruner_client=mock_llm,
            llm_pruner_prompt=PRUNER_PROMPT,
            batch_size=32,
            num_workers=8,
        )
        assert pruner.batch_size == 32
        assert pruner.num_workers == 8
        assert pruner.llm_pruner_prompt == PRUNER_PROMPT
