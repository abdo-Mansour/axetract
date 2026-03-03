import os
import re
import ast
import math
import time
import torch
import threading
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from axetract.utils.json_util import is_schema
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union
from axetract.utils.html_util import merge_html_chunks, extract_visible_xpaths_leaves, merge_xpaths_to_html, clean_html, SmartHTMLProcessor
from axetract.pruner.base_pruner import BasePruner
from axetract.data_types import AXESample, AXEChunk
from axetract.llm.base_client import BaseClient
# ==============================================================================
# 1. STANDALONE HELPER FUNCTIONS (Moved out of class to allow Pickling)
# ==============================================================================

def _longest_common_xpath_prefix(xpaths: Iterable[str]) -> str:
    """Compute longest common xpath prefix."""
    parts_list = []
    for xp in xpaths:
        if not xp: continue
        s = xp if xp.startswith("/") else "/" + xp
        parts_list.append(s.split("/"))

    if not parts_list: return "/"

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
    if s is None: return ""
    return s.replace("'", "\\'")

def _remove_prefix_from_xpath(xpath: str, prefix: str) -> str:
    if xpath is None or xpath == "": return "/"
    if not xpath.startswith("/"): xpath = "/" + xpath
    if prefix == "/": return xpath
    if xpath == prefix: return "/"
    if xpath.startswith(prefix):
        rel = xpath[len(prefix):]
        if rel == "" or not rel.startswith("/"):
            rel = "/" + rel.lstrip("/")
        return rel
    return xpath

def generate_pruner_prompt(xpath_content_pair_ls: List, query: str, prompt_template: str) -> str:
    """
    Standalone version of _promp_gen. 
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
    """
    Worker for _filter.
    Receives: (chunk_content, query, template_string)
    Returns: (chunk_xpaths_object, prompt_string)
    """
    row_content, row_query, prompt_template = args
    
    # 1. Instantiate Processor inside the worker (avoid pickling the object)
    processor = SmartHTMLProcessor() 
    
    # 2. Heavy CPU: Parse HTML
    chunk_xpaths = processor.extract_chunks(row_content)
    
    # 3. Prepare data for prompt generation
    xpath_pairs = [(item['xpath'], item['content']) for item in chunk_xpaths]
    
    # 4. Generate Prompt using STANDALONE function
    prompt = generate_pruner_prompt(xpath_pairs, row_query, prompt_template)
    
    return chunk_xpaths, prompt

def _worker_merge_html(args):
    """
    Worker for _generate_output.
    Receives: (chunks_list, content_fallback)
    Returns: string
    """
    chunks, content = args
    # Import locally to be safe, though util imports are usually fine
    from axetract.utils.html_util import merge_html_chunks
    
    # Heavy CPU: Merge and clean HTML
    merged = merge_html_chunks(chunks, content)
    
    # Optimization: remove newlines here in the worker
    return merged.replace("\n", "")

# ==============================================================================
# 3. CLASS DEFINITION
# ==============================================================================

class AXEPruner(BasePruner):

    def __init__(self,
                llm_pruner_client: BaseClient,
                llm_pruner_prompt: str,
                name: str = "axe_pruner",
                batch_size: int = 16,
                num_workers: int = 4,
                ):
        self.name = name
        self.llm_pruner_client = llm_pruner_client
        self.llm_pruner_prompt = llm_pruner_prompt
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.html_processor = SmartHTMLProcessor()




    def _filter(self, batch: List[AXESample]) -> List[AXESample]:
       
        if len(batch) == 0:
            return batch
        
        
        # Prepare arguments: (content, query, template_string)
        # Note: We pass the template STRING, not self or config.
        template_str = self.llm_pruner_prompt
        worker_args = []
        for sample in batch:
            for chunk in sample.chunks:
                worker_args.append((chunk.content, sample.query, template_str))   

        max_workers = getattr(self, "num_workers", None) or min(32, (os.cpu_count() or 1) * 4)

        # 2. Parallel CPU Execution (HTML Parsing + Prompt Gen)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map returns an iterator, list() consumes it
            results = list(executor.map(_worker_filter_prep, worker_args))

        # Unpack results: results is list of (chunk_xpaths, prompt)
        all_rows_xpaths, prompts = zip(*results)
        all_rows_xpaths = list(all_rows_xpaths)
        prompts = list(prompts)
        
        # 3. Batch GPU Inference (Fast)
        llm_results = self.llm_pruner_client.call_batch(prompts, adapter_name="pruner")
        
        # 4. Process Results (Light CPU work)
        final_pruned_contents = [] # List of list of xpaths

        for response, row_xpaths in zip(llm_results, all_rows_xpaths):
            if not response: 
                final_pruned_contents.append(row_xpaths) 
                continue
            
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            chosen = []
            if match:
                inside = "[" + match.group(1).strip() + "]"
                try:
                    chosen = ast.literal_eval(inside)
                except Exception as e:
                    chosen = []
            
            row_final_list = []
            for idx in chosen:
                if isinstance(idx, int) and 0 <= idx < len(row_xpaths):
                    row_final_list.append(row_xpaths[idx])
            
            final_pruned_contents.append(row_final_list)

        # 5. Reconstruct Samples
        sample_to_pruned_chunks = {}
        for i, sample in enumerate(batch):
            sample_xpath_list = []
            for chunk in sample.chunks:
                chunk_id = chunk.chunkid
                sample_id = chunk_id.split("-")[0]
                pruned_chunk = final_pruned_contents[i]
                sample_xpath_list.append(pruned_chunk)
            sample_to_pruned_chunks[sample_id] = sample_xpath_list
        
        # 6. Merge
        merge_worker_args = []
        for key , value in sample_to_pruned_chunks.items():
            sample = batch[int(key)]
            merge_worker_args.append((value, sample.content))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            new_full_content = list(executor.map(_worker_merge_html, merge_worker_args))
        
        # 7. Update Samples
        for i, sample in enumerate(batch):
            sample.current_html = new_full_content[i]   
            
        return batch

    
    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        filtered_samples = self._filter(samples)
        return filtered_samples