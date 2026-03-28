from __future__ import annotations

import gc
import logging
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, overload

logger = logging.getLogger(__name__)

from pydantic import BaseModel

from axetract.data_types import AXEResult, AXESample, Status
from axetract.extractor.base_extractor import BaseExtractor
from axetract.postprocessor.base_postprocessor import BasePostprocessor
from axetract.preprocessor.base_preprocessor import BasePreprocessor
from axetract.pruner.base_pruner import BasePruner

# Sentinel value to signal stage completion
_SENTINEL = object()


class AXEPipeline:
    """The main orchestrator for the Axetract data extraction process.

    This class coordinates the flow of data through four main stages:
    1. **Preprocessing**: Fetching and cleaning HTML content.
    2. **Pruning**: Using a LoRA-powered LLM to filter out irrelevant DOM nodes.
    3. **Extraction**: Using a LoRA-powered LLM to map HTML content to structured JSON.
    4. **Postprocessing**: Validating schema and performing final cleanup.

    For large batches, the pipeline automatically uses micro-batch pipelining
    to overlap CPU and GPU work across stages. While the GPU runs pruner
    inference on micro-batch N, the CPU can preprocess micro-batch N+1 and
    postprocess micro-batch N-1 concurrently.

    Attributes:
        preprocessor (BasePreprocessor): Component for initial HTML handling.
        pruner (BasePruner): Component for relevance filtering.
        extractor (BaseExtractor): Component for structured data generation.
        postprocessor (BasePostprocessor): Component for results refinement.
        micro_batch_size (int): Number of samples per micro-batch for pipelined
            execution. Smaller values improve overlap but add thread overhead.
    """

    def __init__(
        self,
        preprocessor: BasePreprocessor,
        pruner: BasePruner,
        extractor: BaseExtractor,
        postprocessor: BasePostprocessor,
        micro_batch_size: int = 4,
    ):
        """Initialize the pipeline with its core components.

        Args:
            preprocessor (BasePreprocessor): Component for fetching and cleaning.
            pruner (BasePruner): Component for relevance pruning.
            extractor (BaseExtractor): Component for structured extraction.
            postprocessor (BasePostprocessor): Component for JSON repair and grounding.
            micro_batch_size (int): Micro-batch size for pipelined execution.
                Controls the granularity of CPU/GPU overlap. Default 4.
        """
        self._preprocessor = preprocessor
        self._pruner = pruner
        self._extractor = extractor
        self._postprocessor = postprocessor
        self._micro_batch_size = micro_batch_size

    @staticmethod
    def _free_gpu_cache():
        """Reclaim GPU memory between stages.

        Triggers Python's garbage collector to finalize dead tensor references,
        then returns freed memory blocks to the CUDA allocator. No-op if CUDA
        is not available.
        """
        try:
            import torch

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _read_path_content(path: Path) -> str:
        """Read and return file content as string.

        Args:
            path (Path): Path to the file to read.

        Returns:
            str: The file content decoded as UTF-8.

        Raises:
            ValueError: If the file extension is not .html or .htm.
        """
        if path.suffix.lower() not in (".html", ".htm"):
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                "Only .html and .htm files are supported."
            )
        return path.read_text(encoding="utf-8")

    @classmethod
    def from_config(
        cls, llm_config: Optional[Dict[str, Any]] = None, use_vllm: bool = False
    ) -> "AXEPipeline":
        """Creates a ready-to-use pipeline with default clients, components, and prompts.

        Args:
            llm_config (Optional[Dict[str, Any]]): LLM configuration override.
            use_vllm (bool): Whether to use vLLM for high-throughput serving.

        Returns:
            AXEPipeline: An initialized pipeline instance.
        """
        from axetract.extractor.axe_extractor import AXEExtractor
        from axetract.postprocessor.axe_postprocessor import AXEPostprocessor
        from axetract.preprocessor.axe_preprocessor import AXEPreprocessor
        from axetract.prompts.pruner_prompt import PRUNER_PROMPT
        from axetract.prompts.qa_prompt import QA_PROMPT
        from axetract.prompts.schema_prompt import SCHEMA_PROMPT
        from axetract.pruner.axe_pruner import AXEPruner

        if llm_config is None:
            llm_config = {
                "model_name": "Qwen/Qwen3-0.6B",
                "max_tokens": 512,
                "engine_args": {
                    "gpu_memory_utilization": 0.8,
                    "max_model_len": 8192,
                    "enable_lora": True,
                    "max_loras": 3,
                    "max_lora_rank": 64,
                    "disable_log_stats": True,
                },
                "generation_config": {
                    "temperature": 0.0,
                    "top_p": 0.7,
                },
                "lora_modules": {
                    "pruner": {
                        "path": "abdo-Mansour/AXE-Pruner-Adaptor-Qwen3-0.6b",
                        "temperature": 0.0,
                    },
                    "qa": {
                        "path": "abdo-Mansour/AXE-QA-Adaptor-Qwen3-0.6b",
                        "temperature": 1.0,
                    },
                    "schema": {
                        "path": "abdo-Mansour/AXE-Extractor-Adaptor-Qwen3-0.6b",
                        "temperature": 0.0,
                    },
                },
            }

        if use_vllm:
            from axetract.llm.vllm_client import LocalVLLMClient

            lc = LocalVLLMClient(config=llm_config)
        else:
            from axetract.llm.hf_client import HuggingFaceClient

            lc = HuggingFaceClient(config=llm_config)

        preprocessor = AXEPreprocessor(use_clean_chunker=True, chunk_size=1000)
        pruner = AXEPruner(llm_pruner_client=lc, llm_pruner_prompt=PRUNER_PROMPT)
        extractor = AXEExtractor(
            llm_extractor_client=lc,
            schema_generation_prompt_template=SCHEMA_PROMPT,
            query_generation_prompt_template=QA_PROMPT,
        )
        postprocessor = AXEPostprocessor()

        return cls(
            preprocessor=preprocessor,
            pruner=pruner,
            extractor=extractor,
            postprocessor=postprocessor,
        )

    @overload
    def extract(
        self,
        input_data: Union[str, Path],
        query: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], str, Dict[str, Any]]] = None,
    ) -> AXEResult: ...

    @overload
    def extract(
        self,
        input_data: List[Union[str, Path]],
        query: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], str, Dict[str, Any]]] = None,
    ) -> List[AXEResult]: ...

    def extract(
        self,
        input_data: Union[str, Path, List[Union[str, Path]]],
        query: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], str, Dict[str, Any]]] = None,
    ) -> Union[AXEResult, List[AXEResult]]:
        """Extract structured data from input documents.

        Supports both single documents and multiple documents with the same query.

        Args:
            input_data (Union[str, Path, List[Union[str, Path]]]): URL(s), raw HTML
                content(s), or path(s) to HTML file(s) (.html or .htm). If a Path
                is provided, the file must have a .html or .htm extension and its
                content will be read as HTML.
            query (Optional[str]): Natural language extraction prompt.
            schema (Optional[Union[Type[BaseModel], str, Dict[str, Any]]]): Desired output schema.

        Returns:
            Union[AXEResult, List[AXEResult]]: A single extraction result for single
                input, or a list of results for multiple inputs.
        """
        # Handle list of inputs
        if isinstance(input_data, list):
            batch = []
            for data in input_data:
                if isinstance(data, Path):
                    content_str = self._read_path_content(data)
                    is_url = False
                else:
                    content_str = data
                    is_url = data.strip().startswith(("http://", "https://"))
                batch.append(
                    AXESample(
                        id=str(uuid.uuid4()),
                        content=content_str,
                        is_content_url=is_url,
                        query=query,
                        schema_model=schema,
                    )
                )
            return self.extract_batch(batch)

        # Handle single input
        if isinstance(input_data, Path):
            content_str = self._read_path_content(input_data)
            is_url = False
        else:
            content_str = input_data
            is_url = input_data.strip().startswith(("http://", "https://"))

        sample = AXESample(
            id=str(uuid.uuid4()),
            content=content_str,
            is_content_url=is_url,
            query=query,
            schema_model=schema,
        )

        return self.extract_batch([sample])[0]

    def _format_batch(self, batch: List[Union[AXESample, Dict[str, Any]]]) -> List[AXESample]:
        """Convert dicts to AXESamples if necessary.

        Args:
            batch: Raw batch of samples or dicts.

        Returns:
            List of AXESample objects.
        """
        formatted = []
        for item in batch:
            if isinstance(item, dict):
                input_data = item.get("input_data", "")
                if isinstance(input_data, Path):
                    content_str = self._read_path_content(input_data)
                    is_url = False
                else:
                    content_str = input_data
                    is_url = input_data.strip().startswith(("http://", "https://")) if input_data else False
                formatted.append(
                    AXESample(
                        id=str(item.get("id", uuid.uuid4())),
                        content=content_str,
                        is_content_url=is_url,
                        query=item.get("query"),
                        schema_model=item.get("schema"),
                    )
                )
            else:
                formatted.append(item)
        return formatted

    def _to_results(self, samples: List[AXESample]) -> List[AXEResult]:
        """Convert processed samples to AXEResult objects.

        Args:
            samples: Processed AXESample list.

        Returns:
            List of AXEResult objects.
        """
        return [
            AXEResult(
                id=str(sample.id),
                prediction=sample.prediction or {},
                xpaths=sample.xpaths,
                status=sample.status,
                error=None
                if sample.status == Status.SUCCESS
                else f"Encountered error/pending status: {sample.status}",
            )
            for sample in samples
        ]

    # ──────────────────────────────────────────────────────────────────
    # Sequential Processing (for small batches / single items)
    # ──────────────────────────────────────────────────────────────────

    def _process_sequential(self, batch: List[AXESample]) -> List[AXESample]:
        """Process a batch sequentially through all stages.

        Used for small batches where threading overhead outweighs overlap gains.

        Args:
            batch: Formatted AXESample list.

        Returns:
            Processed AXESample list.
        """
        logger.debug("Sequential processing for batch of size %d", len(batch))

        # 1. Preprocess (Fetch & Chunk)
        logger.debug("Step 1: Running preprocessor...")
        batch = self._preprocessor(batch)
        for i, sample in enumerate(batch):
            chunks = sample.chunks or []
            chunk_summary = [(c.chunkid, len(c.content)) for c in chunks]
            logger.debug(
                "  -> Sample %d after preprocessor: %d chunk(s) -> %s",
                i, len(chunks), chunk_summary,
            )

        # 2. Prune
        if self._pruner:
            logger.debug("Step 2: Running pruner...")
            batch = self._pruner(batch)
            for i, sample in enumerate(batch):
                xpaths = sample.xpaths or []
                logger.debug("  -> Sample %d after pruner: %d xpath(s) -> %s", i, len(xpaths), xpaths)

        # 3. Extract
        self._free_gpu_cache()
        logger.debug("Step 3: Running extractor...")
        batch = self._extractor(batch)
        for i, sample in enumerate(batch):
            logger.debug("  -> Sample %d after extractor: %s", i, sample.prediction)

        # 4. Postprocess
        if self._postprocessor:
            logger.debug("Step 4: Running postprocessor...")
            batch = self._postprocessor(batch)
            for i, sample in enumerate(batch):
                logger.debug("  -> Sample %d after postprocessor: %s", i, sample.prediction)

        return batch

    # ──────────────────────────────────────────────────────────────────
    # Pipelined Processing (for large batches)
    # ──────────────────────────────────────────────────────────────────

    def _process_pipelined(
        self, batch: List[AXESample], micro_batch_size: int
    ) -> List[AXESample]:
        """Process a batch with micro-batch pipelining for CPU/GPU overlap.

        Splits the batch into micro-batches and runs pipeline stages in
        separate threads connected by queues. This way:
        - CPU preprocessing of micro-batch N+1 overlaps with GPU pruning of N
        - CPU postprocessing of micro-batch N overlaps with GPU extraction of N+1
        - The GPU is never fully idle waiting for CPU stages to complete

        The GPU lock in the LLM client ensures inference safety while allowing
        CPU-bound work in other threads to proceed concurrently.

        Args:
            batch: Formatted AXESample list.
            micro_batch_size: Samples per micro-batch.

        Returns:
            Processed AXESample list in original order.
        """
        micro_batches = [
            batch[i : i + micro_batch_size]
            for i in range(0, len(batch), micro_batch_size)
        ]
        num_mbs = len(micro_batches)

        logger.debug(
            "Pipelined processing: %d samples → %d micro-batches of ≤%d",
            len(batch), num_mbs, micro_batch_size,
        )

        # Bounded queues between stages (maxsize=2 limits memory while allowing overlap)
        q_preprocessed = queue.Queue(maxsize=2)
        q_pruned = queue.Queue(maxsize=2)
        q_extracted = queue.Queue(maxsize=2)

        # Results indexed by micro-batch position for ordered reassembly
        results = [None] * num_mbs
        errors = []

        def _preprocess_stage():
            """Stage 1: CPU-bound preprocessing (fetch, clean, chunk)."""
            for mb_idx, mb in enumerate(micro_batches):
                try:
                    logger.debug("[Pipeline] Preprocessing micro-batch %d/%d", mb_idx + 1, num_mbs)
                    processed = self._preprocessor(mb)
                    q_preprocessed.put((mb_idx, processed))
                except Exception as e:
                    logger.error("[Pipeline] Preprocess error on micro-batch %d: %s", mb_idx, e)
                    errors.append(e)
                    q_preprocessed.put((mb_idx, mb))  # Pass through on error
            q_preprocessed.put(_SENTINEL)

        def _prune_stage():
            """Stage 2: CPU prep → GPU inference → CPU post (pruning)."""
            while True:
                item = q_preprocessed.get()
                if item is _SENTINEL:
                    break
                mb_idx, mb = item
                try:
                    if self._pruner:
                        logger.debug("[Pipeline] Pruning micro-batch %d/%d", mb_idx + 1, num_mbs)
                        mb = self._pruner(mb)
                except Exception as e:
                    logger.error("[Pipeline] Pruner error on micro-batch %d: %s", mb_idx, e)
                    errors.append(e)
                q_pruned.put((mb_idx, mb))
            q_pruned.put(_SENTINEL)

        def _extract_stage():
            """Stage 3: CPU prep → GPU inference → CPU post (extraction)."""
            while True:
                item = q_pruned.get()
                if item is _SENTINEL:
                    break
                mb_idx, mb = item
                try:
                    logger.debug("[Pipeline] Extracting micro-batch %d/%d", mb_idx + 1, num_mbs)
                    mb = self._extractor(mb)
                except Exception as e:
                    logger.error("[Pipeline] Extractor error on micro-batch %d: %s", mb_idx, e)
                    errors.append(e)
                q_extracted.put((mb_idx, mb))
            q_extracted.put(_SENTINEL)

        def _postprocess_stage():
            """Stage 4: CPU-bound postprocessing (JSON repair, XPath grounding)."""
            while True:
                item = q_extracted.get()
                if item is _SENTINEL:
                    break
                mb_idx, mb = item
                try:
                    if self._postprocessor:
                        logger.debug(
                            "[Pipeline] Postprocessing micro-batch %d/%d", mb_idx + 1, num_mbs,
                        )
                        mb = self._postprocessor(mb)
                except Exception as e:
                    logger.error("[Pipeline] Postprocess error on micro-batch %d: %s", mb_idx, e)
                    errors.append(e)
                results[mb_idx] = mb

        # Launch all stages as concurrent threads
        threads = [
            threading.Thread(target=_preprocess_stage, name="pipeline-preprocess", daemon=True),
            threading.Thread(target=_prune_stage, name="pipeline-prune", daemon=True),
            threading.Thread(target=_extract_stage, name="pipeline-extract", daemon=True),
            threading.Thread(target=_postprocess_stage, name="pipeline-postprocess", daemon=True),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            logger.warning("[Pipeline] %d error(s) occurred during pipelined processing.", len(errors))

        # Flatten micro-batches back into a single list in original order
        all_samples = []
        for mb in results:
            if mb is not None:
                all_samples.extend(mb)

        logger.debug("Pipelined processing completed: %d samples returned.", len(all_samples))
        return all_samples

    # ──────────────────────────────────────────────────────────────────
    # Public Entry Point
    # ──────────────────────────────────────────────────────────────────

    def extract_batch(self, batch: List[Union[AXESample, Dict[str, Any]]]) -> List[AXEResult]:
        """Main execution flow of the pipeline.

        Accepts a list of AXESample objects OR a list of dictionaries.
        Automatically selects between sequential and pipelined execution:
        - Small batches (≤ micro_batch_size): sequential (no threading overhead)
        - Large batches (> micro_batch_size): pipelined (CPU/GPU overlap)

        Args:
            batch (List[Union[AXESample, Dict[str, Any]]]): Batch of extraction tasks.

        Returns:
            List[AXEResult]: Final processed results.
        """
        # 0. Convert Dicts to AXESamples if necessary
        batch = self._format_batch(batch)
        logger.debug("Starting pipeline processing for batch of size %d", len(batch))

        # Choose execution strategy based on batch size
        if len(batch) <= self._micro_batch_size:
            processed = self._process_sequential(batch)
        else:
            processed = self._process_pipelined(batch, self._micro_batch_size)

        logger.debug("Pipeline processing completed successfully.")
        return self._to_results(processed)
