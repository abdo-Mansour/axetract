import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

from axetract.data_types import AXESample
from axetract.postprocessor.base_postprocessor import BasePostprocessor
from axetract.utils.html_util import (
    build_html_search_index,
    match_against_index,
    normalize_html_text,
)
from axetract.utils.json_util import extract_and_repair_json, is_schema

# ==============================================================================
# WORKER FUNCTIONS (Top-level for Pickling)
# ==============================================================================


def _recursive_exact_extract_indexed(data: Any, index: list) -> Tuple[Any, Any]:
    """Recursively perform matching using a pre-built HTML search index.

    Instead of re-parsing the HTML document for every leaf value, this
    function uses the pre-built index (one parse) for all lookups.

    Args:
        data: Parsed JSON data (dict, list, or scalar).
        index: Pre-built search index from build_html_search_index().

    Returns:
        Tuple of (matched_values, matched_xpaths).
    """
    if isinstance(data, dict):
        result_vals = {}
        result_xpaths = {}
        for k, v in data.items():
            val, xp = _recursive_exact_extract_indexed(v, index)
            result_vals[k] = val
            result_xpaths[k] = xp
        return result_vals, result_xpaths
    elif isinstance(data, list):
        vals = []
        xpaths = []
        for item in data:
            val, xp = _recursive_exact_extract_indexed(item, index)
            vals.append(val)
            xpaths.append(xp)
        return vals, xpaths
    elif data is None:
        return None, None
    else:
        try:
            val_str = str(data)
            best_match = match_against_index(val_str, index)

            val = (
                normalize_html_text(best_match["text"])
                if best_match and best_match.get("found")
                else None
            )
            xp = best_match.get("xpath") if best_match and best_match.get("found") else None

            return val, xp
        except Exception as e:
            return {"__error__": f"[MATCH_ERROR] {e}", "original": data}, None


def _safe_extract_worker(
    response: str, content: str, query: Any, extract_exact: bool
) -> Tuple[Union[Dict[str, Any], str], Optional[Dict[str, Any]]]:
    """Optimized worker that takes raw strings instead of a meta dict.

    Builds the HTML search index ONCE per document, then matches all
    extracted fields against it — eliminating repeated HTML parsing.

    Returns (parsed_response, xpaths).
    """
    try:
        if not response:
            return "", None

        # 1. Parse JSON
        if query is not None and not isinstance(query, str):
            is_schema_query = True
        else:
            query_str = str(query) if query is not None else ""
            is_schema_query = is_schema(query_str) if query_str else False

        parsed_response = extract_and_repair_json(response, not is_schema_query)

        if isinstance(parsed_response, str):
            return parsed_response, None

        # Validate dict
        if not isinstance(parsed_response, dict):
            return {
                "__error__": "[PARSE_ERROR] expected JSON object (dict) from extract_and_repair_json"
            }, None

        # 2. Exact Extraction (parse HTML ONCE, match all fields)
        xpaths = None
        if extract_exact:
            if not content:
                return {
                    "__error__": "[PARSE_ERROR] exact_extraction requested but no content provided"
                }, None

            # Build index ONCE for this document
            index = build_html_search_index(content)
            # Match ALL fields against the pre-built index
            parsed_response, xpaths = _recursive_exact_extract_indexed(parsed_response, index)

        return parsed_response, xpaths

    except Exception as e:
        return {"__error__": f"[PARSE_ERROR] {e}"}, None


# ==============================================================================
# CLASS DEFINITION
# ==============================================================================


class AXEPostprocessor(BasePostprocessor):
    """Optimized PostProcessor for high-throughput batch processing.

    This component handles JSON parsing, repair, and grounded XPath resolution (GXR)
    to map extracted values back to the original document.

    Uses a parse-once indexing strategy: each document's HTML is parsed into a
    search index exactly once, and all extracted fields are matched against that
    index. This eliminates the O(fields × parse_cost) bottleneck.

    Attributes:
        name (str): Component name.
        exact_extraction (bool): Whether to perform fuzzy matching to find source XPaths.
    """

    def __init__(self, name: str = "axe_postprocessor", exact_extraction: bool = True):
        """Initialize the postprocessor.

        Args:
            name (str): Component name.
            exact_extraction (bool): Enable grounded XPath resolution.
        """
        super().__init__(name=name)
        self._exact_extraction = exact_extraction

    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        """Clean, repair, and ground a batch of extraction samples.

        Args:
            samples (List[AXESample]): Samples with raw LLM predictions.

        Returns:
            List[AXESample]: Samples with structured predictions and XPaths.
        """
        if not samples:
            return samples

        n_items = len(samples)
        # Leave one core free for system stability
        n_workers = max(1, (os.cpu_count() or 2) - 1)

        # Use ThreadPoolExecutor to avoid expensive pickling of large HTML strings.
        # The CPU-bound work (SequenceMatcher) partially releases the GIL, and
        # the index-based approach makes each worker much faster than before.
        executor_cls = ThreadPoolExecutor

        # Prepare flags generator
        extract_flags = [self._exact_extraction] * n_items

        # We assume sample.prediction holds the raw LLM string generated by the extractor
        responses = [
            s.prediction if isinstance(s.prediction, str) else str(s.prediction) for s in samples
        ]
        queries = [s.query or s.schema_model for s in samples]

        # We perform exact_extraction matching using the current HTML processing output
        contents = [s.current_html for s in samples]

        # EXECUTE IN PARALLEL
        with executor_cls(max_workers=n_workers) as ex:
            parsed_results = list(
                ex.map(_safe_extract_worker, responses, contents, queries, extract_flags)
            )

        # Re-assemble results in the main process
        for sample, (parsed, xpaths) in zip(samples, parsed_results):
            sample.prediction = parsed
            sample.xpaths = xpaths

        return samples
