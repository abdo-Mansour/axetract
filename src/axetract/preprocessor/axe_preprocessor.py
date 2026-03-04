from __future__ import annotations
from typing import Any, Dict, List, Union
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from axetract.utils.html_util import fetch_content, clean_html, chunk_html_content
from axetract.preprocessor.base_preprocessor import BasePreprocessor
from axetract.data_types import AXESample, AXEChunk

def _chunk_worker(args: tuple) -> Dict[str, Any]:
    sample, config, idx = args
    cleaned_text = clean_html(
        html_content=sample.content,
        extra_remove_tags=config.extra_remove_tags,
        strip_attrs=config.strip_attrs,
        strip_links=config.strip_links,
        keep_tags=config.keep_tags,
        use_clean_rag=config.use_clean_rag)
    
    try:
        if not cleaned_text:
            return {'doc_id': idx, 'chunks': [{'chunkid': f"{idx}-err", 'chunkcontent': '[Chunk Worker ERROR] empty content or fetch failed'}]}
        if config.disable_chunking:
            chunks = [cleaned_text]
        else:
            chunks = chunk_html_content(html_content=cleaned_text,
                                        max_tokens=config.chunk_size,
                                        is_clean=config.use_clean_chunker,
                                        attr_cutoff_len=config.attr_cutoff_len)
        
        chunks_list = [{'chunkid': f"{idx}-{i+1}", 'chunkcontent': c} for i, c in enumerate(chunks)]
        return {'doc_id': idx, 'chunks': chunks_list}
    except Exception as e:
        tb = traceback.format_exc()
        err_payload = {
            "doc_id": idx,
            "chunks": [
                {
                    "chunkid": f"{idx}-err",
                    "chunkcontent": f"[ERROR {type(e).__name__}] {e}\n{tb}"
                }
            ]
        }
        return idx, err_payload


class AXEPreprocessor(BasePreprocessor):
    
    def __init__(self,
                 name: str = "AXEPreprocessor",
                 fetch_workers: int = 1,
                 cpu_workers: int = 1,
                 extra_remove_tags: List[str] = None,
                 strip_attrs: bool = True,
                 strip_links: bool = True,
                 keep_tags: bool = False,
                 use_clean_rag: bool = True,
                 use_clean_chunker: bool = True,
                 chunk_size: int = 1000,
                 attr_cutoff_len: int = 100,
                 disable_chunking: bool = False):
        super().__init__(name)
        self.fetch_workers = fetch_workers
        self.cpu_workers = cpu_workers

        self.extra_remove_tags = extra_remove_tags
        self.strip_attrs = strip_attrs
        self.strip_links = strip_links
        self.keep_tags = keep_tags
        self.use_clean_rag = use_clean_rag
        self.use_clean_chunker = use_clean_chunker
        self.chunk_size = chunk_size
        self.attr_cutoff_len = attr_cutoff_len
        self.disable_chunking = disable_chunking


    def __call__(self, batch: List[AXESample]) -> List[AXESample]:
        
        if isinstance(batch, AXESample):
            batch = [batch]
        
        n = len(batch)
        if n == 0:
            return []

        def _quick_fetch(sample: AXESample) -> AXESample:
            if sample.is_content_url:
                try:
                    fetched = fetch_content(sample.content)
                    sample.content = fetched
                    sample.is_content_url = False
                    return sample
                except Exception as e:
                    sample.content = f"[Fetch ERROR] {e}"
                    sample.is_content_url = False
                    return sample
            else:
                return sample
        # Fetching the content if there are any URLs
        with ThreadPoolExecutor(max_workers=min(self.fetch_workers, max(1, n))) as tpool:
            batch = list(tpool.map(_quick_fetch, batch))
        

        results = [None] * n
        # choose executor class and max_workers
        use_processes = bool(self.cpu_workers and self.cpu_workers > 1)
        ExecutorCls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        if use_processes:
            max_workers = min(self.cpu_workers or n, max(1, n))
        else:
            # for IO-bound thread fallback, use fetch_workers if provided else at most n
            max_workers = min(self.fetch_workers or n, max(1, n))

        # prepare enumerated args (Sample, config, index)
        items = [(batch[i],self,i) for i in range(n)]

        results: List[Dict] = [None] * n

        with ExecutorCls(max_workers=max_workers) as ex:
            for idx, res in enumerate(ex.map(_chunk_worker, items)):
                results[idx] = res

        for i, res in enumerate(results):
            batch[i].chunks = [AXEChunk(chunkid=chunk['chunkid'], content=chunk['chunkcontent']) for chunk in res['chunks']]
            
        return batch


        