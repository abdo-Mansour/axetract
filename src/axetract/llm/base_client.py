import os
import random
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import List, Iterable, Optional, Any, Callable, Dict, Union



class BaseClient(ABC):
    """Abstract base class for calling LLMs across any backend."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def call_api(self, prompt: str, adapter_name: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    def call_batch(
        self,
        prompts: Iterable[str],
        max_workers: int = 8,
        chunk_size: Optional[int] = None,
        raise_on_error: bool = False,
        adapter_name: Optional[str] = None,
        per_result_callback: Optional[Callable[[int, Optional[str], Optional[Exception]], Any]] = None,
        **call_api_kwargs,
    ) -> List[Optional[str]]:
        """
        Default threaded batching.
        NOTE: Local models (HF/vLLM) should override this to use proper native batching.
        """
        prompts = list(prompts)
        results: List[Optional[str]] = [None] * len(prompts)

        def _submit_range(start_idx: int, end_idx: int):
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i in range(start_idx, end_idx):
                    fut = ex.submit(self.call_api, prompts[i], adapter_name=adapter_name, **call_api_kwargs)
                    futures[fut] = i
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        res = fut.result()
                        results[idx] = res
                        if per_result_callback:
                            per_result_callback(idx, res, None)
                    except Exception as exc:
                        if per_result_callback:
                            per_result_callback(idx, None, exc)
                        if raise_on_error:
                            raise
                        results[idx] = None

        if chunk_size is None or chunk_size <= 0:
            _submit_range(0, len(prompts))
        else:
            for start in range(0, len(prompts), chunk_size):
                end = min(start + chunk_size, len(prompts))
                _submit_range(start, end)

        return results

