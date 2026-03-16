import json
import logging
from typing import List

from pydantic import BaseModel

from axetract.data_types import AXESample, Status
from axetract.extractor.base_extractor import BaseExtractor
from axetract.llm.base_client import BaseClient
from axetract.utils.json_util import is_schema

logger = logging.getLogger(__name__)


class AXEExtractor(BaseExtractor):
    """Component for extracting structured data from HTML using LLMs.

    Attributes:
        llm_extractor_client (BaseClient): The LLM client used for extraction.
        schema_prompt_template (str): Template for schema-based extraction prompts.
        query_prompt_template (str): Template for natural language query prompts.
        name (str): Component name.
        batch_size (int): Processing batch size.
        num_workers (int): Number of parallel workers.
    """

    def __init__(
        self,
        llm_extractor_client: BaseClient,
        schema_generation_prompt_template: str,
        query_generation_prompt_template: str,
        name: str = "axe_extractor",
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """Initialize the extractor.

        Args:
            llm_extractor_client (BaseClient): LLM client.
            schema_generation_prompt_template (str): Schema prompt template.
            query_generation_prompt_template (str): Query prompt template.
            name (str): Component name.
            batch_size (int): Batch size.
            num_workers (int): Parallel workers.
        """
        self.llm_extractor_client = llm_extractor_client
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.schema_prompt_template = schema_generation_prompt_template
        self.query_prompt_template = query_generation_prompt_template

    def _generate_output(self, samples: List[AXESample]) -> List[AXESample]:

        def build_prompt(data):
            query = data.query or data.schema_model
            content = data.current_html

            # Convert Query/Schema to appropriate string if it is a dictionary or Pydantic model
            if query is not None and not isinstance(query, str):
                if isinstance(query, dict):
                    query = json.dumps(query)
                elif isinstance(query, type) and issubclass(query, BaseModel):
                        # For Pydantic V2 use model_json_schema, for V1 use schema_json
                        if hasattr(query, "model_json_schema"):
                            query = json.dumps(query.model_json_schema())
                        elif hasattr(query, "schema_json"):
                            query = query.schema_json()

            if is_schema(query):
                return self.schema_prompt_template.format(query=query, content=content)
            else:
                return self.query_prompt_template.format(query=query, content=content)

        prompts = [build_prompt(sample) for sample in samples]
        queries = [sample.query or sample.schema_model for sample in samples]

        # Storage for split batches
        qa_indices = []
        qa_prompts = []

        schema_indices = []
        schema_prompts = []

        # 1. Split based on Query Type
        for idx, (q, p) in enumerate(zip(queries, prompts)):
            if is_schema(q):
                schema_indices.append(idx)
                schema_prompts.append(p)
            else:
                qa_indices.append(idx)
                qa_prompts.append(p)

        # Holder for final results in original order
        final_responses = [None] * len(prompts)

        # 2. Run QA Batch (Adapter: "qa")
        if qa_prompts:
            logger.debug("Processing %d QA queries...", len(qa_prompts))
            for idx, (orig_idx, q) in enumerate(zip(qa_indices, queries)):
                if not is_schema(q):
                    logger.debug("  [QA] sample %d query: %s", orig_idx, q)
                    logger.debug("  [QA] sample %d prompt: %s", orig_idx, qa_prompts[idx])
            qa_responses = self.llm_extractor_client.call_batch(qa_prompts, adapter_name="qa")

            for original_idx, response in zip(qa_indices, qa_responses):
                logger.debug("  [QA] sample %d response: %s", original_idx, response)
                final_responses[original_idx] = response

        # 3. Run Schema Batch (Adapter: "schema")
        if schema_prompts:
            logger.debug("Processing %d Schema queries...", len(schema_prompts))
            for idx, (orig_idx, q) in enumerate(zip(schema_indices, queries)):
                if is_schema(q):
                    logger.debug("  [Schema] sample %d schema: %s", orig_idx, q)
                    logger.debug("  [Schema] sample %d prompt: %s", orig_idx, schema_prompts[idx])
            schema_responses = self.llm_extractor_client.call_batch(
                schema_prompts, adapter_name="schema"
            )

            for original_idx, response in zip(schema_indices, schema_responses):
                logger.debug("  [Schema] sample %d response: %s", original_idx, response)
                final_responses[original_idx] = response

        for sample, response in zip(samples, final_responses):
            sample.prediction = response
            sample.status = Status.SUCCESS if response is not None else Status.FAILED
            logger.debug("  [Extractor] sample %s final prediction: %s", sample.id, response)
        return samples

    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        """Run the extraction process on a batch of samples.

        Args:
            samples (List[AXESample]): Input samples with clean HTML.

        Returns:
            List[AXESample]: Samples with LLM-generated predictions.
        """
        # Step 3: Generate (Optimized Parallel)
        generated_samples = self._generate_output(samples)

        return generated_samples
