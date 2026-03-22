import ast
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List

from axetract.data_types import AXESample
from axetract.llm.base_client import BaseClient
from axetract.pruner.base_pruner import BasePruner
from axetract.utils.html_util import (
    SmartHTMLProcessor,
    merge_html_chunks,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. STANDALONE HELPER FUNCTIONS (Moved out of class to allow Pickling)
# ==============================================================================


def _longest_common_xpath_prefix(xpaths: Iterable[str]) -> str:
    """Compute longest common xpath prefix."""
    parts_list = []
    for xp in xpaths:
        if not xp:
            continue
        s = xp if xp.startswith("/") else "/" + xp
        parts_list.append(s.split("/"))

    if not parts_list:
        return "/"

    common = []
    for segs in zip(*parts_list):
        if all(seg == segs[0] for seg in segs):
            common.append(segs[0])
        else:
            break

    if not common or (len(common) == 1 and common[0] == ""):
        return "/"
    prefix = "/".join(common)
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix


def _escape_single_quotes(s: str) -> str:
    if s is None:
        return ""
    return s.replace("'", "\\'")


def _remove_prefix_from_xpath(xpath: str, prefix: str) -> str:
    if xpath is None or xpath == "":
        return "/"
    if not xpath.startswith("/"):
        xpath = "/" + xpath
    if prefix == "/":
        return xpath
    if xpath == prefix:
        return "/"
    if xpath.startswith(prefix):
        rel = xpath[len(prefix) :]
        if rel == "" or not rel.startswith("/"):
            rel = "/" + rel.lstrip("/")
        return rel
    return xpath


def generate_pruner_prompt(xpath_content_pair_ls: List, query: str, prompt_template: str) -> str:
    """Standalone version of _promp_gen.

    Accepts prompt_template string directly instead of config object.
    """
    normalized = []
    for pair in xpath_content_pair_ls:
        if pair is None:
            normalized.append(("", ""))
            continue
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            xpath, text = pair[0] or "", pair[1] or ""
        elif isinstance(pair, dict):
            xpath = pair.get("xpath", "") or pair.get("0", "") or ""
            text = pair.get("content", "") or pair.get("1", "") or ""
        else:
            try:
                xpath = str(pair[0]) if getattr(pair, "__len__", None) and len(pair) >= 1 else ""
                text = str(pair[1]) if getattr(pair, "__len__", None) and len(pair) >= 2 else ""
            except Exception:
                xpath, text = "", str(pair)
        normalized.append((xpath, text))

    xpaths_for_prefix = [xp for xp, _ in normalized if xp]
    prefix = _longest_common_xpath_prefix(xpaths_for_prefix)

    lines = []
    lines.append(f"The entire chunk is under: '{_escape_single_quotes(prefix)}'")

    for idx, (xp, txt) in enumerate(normalized):
        rel = _remove_prefix_from_xpath(xp, prefix)
        rel_escaped = _escape_single_quotes(rel)
        txt_escaped = _escape_single_quotes(txt)
        if not rel_escaped.startswith("/"):
            rel_escaped = "/" + rel_escaped
        lines.append(f"{idx} ('{rel_escaped}', '{txt_escaped}')")

    full_content = "\n".join(lines)
    prompt = prompt_template.format(query=query, content=full_content)
    return prompt


# ==============================================================================
# 2. WORKER FUNCTIONS (Must be top-level for Multiprocessing)
# ==============================================================================


def _worker_filter_prep(args):
    """Worker for _filter.

    Receives: (chunk_content, query, template_string)
    Returns: (chunk_xpaths_object, prompt_string)
    """
    row_content, row_query, prompt_template = args

    # 1. Instantiate Processor inside the worker (avoid pickling the object)
    processor = SmartHTMLProcessor()

    # 2. Heavy CPU: Parse HTML
    chunk_xpaths = processor.extract_chunks(row_content)

    # 3. Prepare data for prompt generation
    xpath_pairs = [(item["xpath"], item["content"]) for item in chunk_xpaths]

    # 4. Generate Prompt using STANDALONE function
    prompt = generate_pruner_prompt(xpath_pairs, row_query, prompt_template)

    return chunk_xpaths, prompt


def _worker_merge_html(args):
    """Worker for _generate_output.

    Receives: (chunks_list, content_fallback)
    Returns: string.
    """
    chunks, content = args
    # Import locally to be safe, though util imports are usually fine

    # Heavy CPU: Merge and clean HTML
    merged = merge_html_chunks(chunks, content)

    # Optimization: remove newlines here in the worker
    return merged.replace("\n", "")


# ==============================================================================
# 3. CLASS DEFINITION
# ==============================================================================


class AXEPruner(BasePruner):
    """Component for pruning HTML content to keep only relevant nodes using small LLM.

    Attributes:
        llm_pruner_client (BaseClient): The LLM client used for pruning.
        llm_pruner_prompt (str): The prompt template for the pruner.
        name (str): Component name.
        batch_size (int): Processing batch size.
        num_workers (int): Number of parallel workers for CPU-bound tasks.
        html_processor (SmartHTMLProcessor): Internal processor for HTML manipulation.
    """

    def __init__(
        self,
        llm_pruner_client: BaseClient,
        llm_pruner_prompt: str,
        name: str = "axe_pruner",
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """Initialize the pruner.

        Args:
            llm_pruner_client (BaseClient): LLM client.
            llm_pruner_prompt (str): Pruner prompt template.
            name (str): Component name.
            batch_size (int): Batch size.
            num_workers (int): Parallel workers.
        """
        self.name = name
        self.llm_pruner_client = llm_pruner_client
        self.llm_pruner_prompt = llm_pruner_prompt
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _filter(self, batch: List[AXESample]) -> List[AXESample]:
        """Identify and filter relevant HTML chunks from a batch of samples.

        Optimized flow: reuses a single thread pool for both HTML parsing
        and merging phases, and uses a pre-compiled regex for response parsing.

        Args:
            batch (List[AXESample]): Input samples with chunks populated.

        Returns:
            List[AXESample]: Samples with current_html set to pruned content.
        """
        if len(batch) == 0:
            logger.debug("_filter received empty batch, skipping.")
            return batch

        logger.debug("[Pruner] Starting _filter on %d samples.", len(batch))

        # Prepare arguments: (content, query, template_string)
        template_str = self.llm_pruner_prompt
        worker_args = []
        for sample in batch:
            for chunk in sample.chunks:
                worker_args.append((chunk.content, sample.query or sample.schema_model, template_str))

        max_workers = getattr(self, "num_workers", None) or min(32, (os.cpu_count() or 1) * 4)
        total_chunks = len(worker_args)
        logger.debug("[Pruner] Preparing %d chunk(s) across %d sample(s) with %d worker(s).",
                    total_chunks, len(batch), max_workers)

        # Use a single thread pool for both CPU-heavy phases
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Phase 1: Parallel CPU — HTML Parsing + Prompt Generation
            results = list(executor.map(_worker_filter_prep, worker_args))
            logger.debug("[Pruner] HTML parsing + prompt generation complete for %d chunk(s).", total_chunks)

            # Unpack results: results is list of (chunk_xpaths, prompt)
            all_rows_xpaths, prompts = zip(*results) if results else ([], [])
            all_rows_xpaths = list(all_rows_xpaths)
            prompts = list(prompts)

            # Phase 2: GPU Inference
            logger.debug("[Pruner] Sending %d prompt(s) to LLM (adapter=pruner).", len(prompts))
            llm_results = self.llm_pruner_client.call_batch(prompts, adapter_name="pruner")
            logger.debug("[Pruner] Raw LLM responses: %s", llm_results)

            # Phase 3: Parse LLM Responses (light CPU work)
            # Pre-compiled regex for index list extraction
            _INDEX_LIST_RE = re.compile(r"\[(.*?)\]", re.DOTALL)
            final_pruned_contents = []

            for i, (response, row_xpaths) in enumerate(zip(llm_results, all_rows_xpaths)):
                if not response:
                    logger.warning("[Pruner] Empty LLM response for chunk %d — keeping full chunk.", i)
                    final_pruned_contents.append(row_xpaths)
                    continue

                match = _INDEX_LIST_RE.search(response)
                chosen = []
                if match:
                    inside = "[" + match.group(1).strip() + "]"
                    try:
                        chosen = ast.literal_eval(inside)
                    except Exception:
                        logger.warning("[Pruner] Failed to parse index list for chunk %d: %r", i, inside)
                        chosen = []
                else:
                    logger.warning("[Pruner] No index list found in LLM response for chunk %d.", i)

                row_final_list = [
                    row_xpaths[idx] for idx in chosen
                    if isinstance(idx, int) and 0 <= idx < len(row_xpaths)
                ]

                logger.debug("[Pruner] Chunk %d: kept %d/%d xpath node(s).",
                             i, len(row_final_list), len(row_xpaths))
                final_pruned_contents.append(row_final_list)

            # Phase 4: Reconstruct Samples
            sample_to_pruned_mini_chunks = {}
            chunk_idx = 0
            for i, sample in enumerate(batch):
                sample_xpath_list = []
                for chunk in sample.chunks:
                    chunk_id = chunk.chunkid
                    sample_id = chunk_id.split("-")[0]
                    pruned_chunk = final_pruned_contents[chunk_idx]
                    sample_xpath_list.append(pruned_chunk)
                    chunk_idx += 1
                sample_to_pruned_mini_chunks[sample_id] = sample_xpath_list
            logger.debug("[Pruner] Sample → pruned mini-chunks map: %s", sample_to_pruned_mini_chunks)

            # Phase 5: Parallel CPU — Merge HTML (reuse pool)
            merge_worker_args = []
            for key, value in sample_to_pruned_mini_chunks.items():
                sample = batch[int(key)]
                merge_worker_args.append((value, sample.content))

            logger.debug("[Pruner] Merging HTML for %d sample(s).", len(merge_worker_args))
            new_full_content = list(executor.map(_worker_merge_html, merge_worker_args))

        # Phase 6: Update Samples
        for i, sample in enumerate(batch):
            before = len(sample.content) if sample.content else 0
            after = len(new_full_content[i]) if new_full_content[i] else 0
            logger.debug("[Pruner] Sample %d: HTML size %d → %d chars (%.1f%% reduction).",
                        i, before, after, 100 * (1 - after / before) if before else 0)
            logger.debug("Sample %d: HTML content: %s", i, new_full_content[i])
            sample.current_html = new_full_content[i]

        logger.debug("[Pruner] _filter complete. Returning %d sample(s).", len(batch))
        return batch

    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        """Execute the pruning process on a list of samples.

        Args:
            samples (List[AXESample]): Input samples.

        Returns:
            List[AXESample]: Pruned samples.
        """
        logger.debug("[Pruner] __call__ received %d sample(s).", len(samples))
        filtered_samples = self._filter(samples)
        logger.debug("[Pruner] __call__ done. %d sample(s) returned.", len(filtered_samples))
        return filtered_samples
