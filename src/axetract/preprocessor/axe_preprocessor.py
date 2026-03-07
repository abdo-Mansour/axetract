from __future__ import annotations

import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List

from axetract.data_types import AXEChunk, AXESample
from axetract.preprocessor.base_preprocessor import BasePreprocessor
from axetract.utils.html_util import chunk_html_content, clean_html, fetch_content


def _chunk_worker(args: tuple) -> Dict[str, Any]:
    sample, config, idx = args
    try:
        cleaned_text = clean_html(
            html_content=sample.content,
            extra_remove_tags=config.extra_remove_tags,
            strip_attrs=config.strip_attrs,
            strip_links=config.strip_links,
            keep_tags=config.keep_tags,
            use_clean_rag=config.use_clean_rag,
        )

        if not cleaned_text:
            return {
                "doc_id": idx,
                "chunks": [
                    {
                        "chunkid": f"{idx}-err",
                        "chunkcontent": "[Chunk Worker ERROR] empty content or fetch failed",
                    }
                ],
            }
        if config.disable_chunking:
            chunks = [cleaned_text]
        else:
            chunks = chunk_html_content(
                html_content=cleaned_text,
                max_tokens=config.chunk_size,
                is_clean=config.use_clean_chunker,
                attr_cutoff_len=config.attr_cutoff_len,
            )

        chunks_list = [
            {"chunkid": f"{idx}-{i + 1}", "chunkcontent": c} for i, c in enumerate(chunks)
        ]
        return {"doc_id": idx, "chunks": chunks_list}
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "doc_id": idx,
            "chunks": [
                {"chunkid": f"{idx}-err", "chunkcontent": f"[ERROR {type(e).__name__}] {e}\n{tb}"}
            ],
        }


class AXEPreprocessor(BasePreprocessor):
    """Component for fetching and chunking HTML content.

    Attributes:
        fetch_workers (int): Number of parallel threads for fetching URLs.
        cpu_workers (int): Number of parallel processes/threads for cleaning and chunking.
        extra_remove_tags (List[str], optional): Additional HTML tags to remove.
        strip_attrs (bool): Whether to remove all tag attributes.
        strip_links (bool): Whether to replace <a> tags with text.
        keep_tags (bool): Whether to preserve HTML tags in the output.
        use_clean_rag (bool): Whether to use htmlrag for cleaning.
        use_clean_chunker (bool): Whether the chunker should expect clean HTML.
        chunk_size (int): Targeted token size for each chunk.
        attr_cutoff_len (int): Length threshold for attribute retention.
        disable_chunking (bool): Whether to skip the chunking step.
    """

    def __init__(
        self,
        name: str = "AXEPreprocessor",
        fetch_workers: int = mp.cpu_count(),
        cpu_workers: int = mp.cpu_count(),
        extra_remove_tags: List[str] | None = ["header", "footer"],
        strip_attrs: bool = True,
        strip_links: bool = True,
        keep_tags: bool = True,
        use_clean_rag: bool = True,
        use_clean_chunker: bool = True,
        chunk_size: int = 2000,
        attr_cutoff_len: int = 5,
        disable_chunking: bool = False,
    ):
        """Initialize the preprocessor.

        Args:
            name (str): Component name.
            fetch_workers (int): Fetching thread count.
            cpu_workers (int): Cleaning process count.
            extra_remove_tags (List[str], optional): Tags to strip.
            strip_attrs (bool): Strip attributes flag.
            strip_links (bool): Strip <a> tags flag.
            keep_tags (bool): Keep HTML tags flag.
            use_clean_rag (bool): Use htmlrag flag.
            use_clean_chunker (bool): Clean chunker flag.
            chunk_size (int): Chunk token limit.
            attr_cutoff_len (int): Attribute length limit.
            disable_chunking (bool): Disable chunking flag.
        """
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
        """Fetch, clean, and chunk a batch of samples.

        Args:
            batch (List[AXESample]): Input samples (URLs or raw HTML).

        Returns:
            List[AXESample]: Samples with chunks populated.
        """
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
        executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        if use_processes:
            max_workers = min(self.cpu_workers or n, max(1, n))
        else:
            # for IO-bound thread fallback, use fetch_workers if provided else at most n
            max_workers = min(self.fetch_workers or n, max(1, n))

        # prepare enumerated args (Sample, config, index)
        items = [(batch[i], self, i) for i in range(n)]

        results: List[Dict] = [None] * n

        with executor_cls(max_workers=max_workers) as ex:
            for idx, res in enumerate(ex.map(_chunk_worker, items)):
                results[idx] = res

        for i, res in enumerate(results):
            batch[i].chunks = [
                AXEChunk(chunkid=chunk["chunkid"], content=chunk["chunkcontent"])
                for chunk in res["chunks"]
            ]

        return batch
