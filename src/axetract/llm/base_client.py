from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional


class BaseClient(ABC):
    """Abstract base class for calling LLMs across any backend."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize the LLM client.

        Args:
            config (Optional[dict]): Backend-specific configuration.
        """
        self.config = config or {}

    @abstractmethod
    def call_api(self, prompt: str, adapter_name: Optional[str] = None, **kwargs) -> str:
        """Call a single LLM completion API.

        Args:
            prompt (str): Input text.
            adapter_name (Optional[str]): Name of the LoRA adapter to use.
            **kwargs: Generation parameter overrides.

        Returns:
            str: The generated text.
        """
        raise NotImplementedError

    def call_batch(
        self,
        prompts: Iterable[str],
        max_workers: int = 8,
        chunk_size: Optional[int] = None,
        raise_on_error: bool = False,
        adapter_name: Optional[str] = None,
        per_result_callback: Optional[
            Callable[[int, Optional[str], Optional[Exception]], Any]
        ] = None,
        **call_api_kwargs,
    ) -> List[Optional[str]]:
        """Process a batch of prompts using threaded parallelism.

        NOTE: Local models (HF/vLLM) should override this to use native engine batching.

        Args:
            prompts (Iterable[str]): Batch of input texts.
            max_workers (int): ThreadPool worker count.
            chunk_size (Optional[int]): If set, processes in sub-batches.
            raise_on_error (bool): Whether to abort on first API error.
            adapter_name (Optional[str]): LoRA adapter name.
            per_result_callback (Optional[Callable]): Hook called for each result.
            **call_api_kwargs: Common parameters passed to call_api.

        Returns:
            List[Optional[str]]: List of completions in matching order.
        """
        prompts = list(prompts)
        results: List[Optional[str]] = [None] * len(prompts)

        def _submit_range(start_idx: int, end_idx: int):
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i in range(start_idx, end_idx):
                    fut = ex.submit(
                        self.call_api, prompts[i], adapter_name=adapter_name, **call_api_kwargs
                    )
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
