from __future__ import annotations
import uuid
from typing import List, Union, Optional, Type, Any, Dict
from pydantic import BaseModel

from axetract.preprocessor.base_preprocessor import BasePreprocessor
from axetract.pruner.base_pruner import BasePruner
from axetract.extractor.base_extractor import BaseExtractor
from axetract.postprocessor.base_postprocessor import BasePostprocessor

from axetract.data_types import AXESample, AXEResult, Status


class AXEPipeline:
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        pruner: BasePruner,
        extractor: BaseExtractor,
        postprocessor: BasePostprocessor,
    ):
        self.preprocessor = preprocessor
        self.pruner = pruner
        self.extractor = extractor
        self.postprocessor = postprocessor

    @classmethod
    def from_config(
        cls, llm_config: Optional[Dict[str, Any]] = None, use_vllm: bool = False
    ) -> "AXEPipeline":
        """
        Creates a ready-to-use pipeline with default clients, components, and prompts.
        """
        from axetract.preprocessor.axe_preprocessor import AXEPreprocessor
        from axetract.pruner.axe_pruner import AXEPruner
        from axetract.extractor.axe_extractor import AXEExtractor
        from axetract.postprocessor.axe_postprocessor import AXEPostprocessor
        from axetract.prompts.pruner_prompt import PRUNER_PROMPT
        from axetract.prompts.qa_prompt import QA_PROMPT
        from axetract.prompts.schema_prompt import SCHEMA_PROMPT

        if llm_config is None:
            llm_config = {
                "model_name": "Qwen/Qwen3-0.6B",
                "max_tokens": 1024,
                "engine_args": {
                    "gpu_memory_utilization": 0.8,
                    "max_model_len": 1024,
                    "enable_lora": True,
                    "max_loras": 3,
                    "max_lora_rank": 64,
                    "disable_log_stats": True,
                },
                "generation_config": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                "lora_modules": {
                    "pruner": {
                        "path": "abdo-Mansour/Pruner_Adaptor_Qwen_3_FINAL_EXTRA",
                        "temperature": 0.0,
                    },
                    "qa": {
                        "path": "abdo-Mansour/Extractor_Adaptor_Qwen3_Final",
                        "temperature": 0.0,
                    },
                    "schema": {
                        "path": "abdo-Mansour/Extractor_Adaptor_Qwen3_Final",
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

        preprocessor = AXEPreprocessor(use_clean_chunker=True)
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

    def process(
        self,
        input_data: str,
        query: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], str, Dict[str, Any]]] = None,
    ) -> AXEResult:
        # 1. Create sample
        sample = AXESample(
            id=str(uuid.uuid4()),
            content=input_data,
            # TODO: might need to do actual util function to check if it is url
            is_content_url=input_data.strip().startswith(("http://", "https://")),
            query=query,
            schema_model=schema,
        )

        return self.process_batch([sample])[0]

    def process_batch(self, batch: List[AXESample]) -> List[AXEResult]:
        """
        Main execution flow of the pipeline.
        Calls the components in sequence: Preprocessor -> Pruner -> Extractor -> Postprocessor.
        """
        # 1. Preprocess (Fetch & Chunk)
        batch = self.preprocessor(batch)

        # 2. Prune (Optional step to reduce tokens)
        if self.pruner:
            batch = self.pruner(batch)

        # 3. Extract (The core extraction logic)
        batch = self.extractor(batch)

        # 4. Postprocess (Optional cleanup)
        if self.postprocessor:
            batch = self.postprocessor(batch)

        # 5. Convert to Results
        results = [
            AXEResult(
                id=str(sample.id),
                prediction=sample.prediction or {},
                xpaths=sample.xpaths,
                status=sample.status,
                error=None
                if sample.status == Status.SUCCESS
                else f"Encountered error/pending status: {sample.status}",
            )
            for sample in batch
        ]

        return results