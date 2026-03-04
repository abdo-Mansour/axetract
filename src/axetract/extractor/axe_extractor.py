import ast
import time
import re
import os
import math
import torch
import threading
import concurrent.futures
from axetract.utils.json_util import is_schema
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union
from axetract.utils.html_util import merge_html_chunks, extract_visible_xpaths_leaves, merge_xpaths_to_html, clean_html, SmartHTMLProcessor
from axetract.extractor.base_extractor import BaseExtractor
from axetract.data_types import AXESample, Status
from axetract.llm.base_client import BaseClient

class AXEExtractor(BaseExtractor):

    def __init__(self, 
                llm_extractor_client: BaseClient, 
                schema_generation_prompt_template: str,
                query_generation_prompt_template: str,
                name: str = "axe_extractor", 
                batch_size: int = 16, 
                num_workers: int = 4):

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
                import json
                if isinstance(query, dict):
                    query = json.dumps(query)
                else:
                    from pydantic import BaseModel
                    if isinstance(query, type) and issubclass(query, BaseModel):
                        # For Pydantic V2 use model_json_schema, for V1 use schema_json
                        if hasattr(query, 'model_json_schema'):
                            query = json.dumps(query.model_json_schema())
                        elif hasattr(query, 'schema_json'):
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
            # print(f"Processing {len(qa_prompts)} QA queries...")
            qa_responses = self.llm_extractor_client.call_batch(qa_prompts, adapter_name="qa")
            
            # Map back to original indices
            for original_idx, response in zip(qa_indices, qa_responses):
                final_responses[original_idx] = response

        # 3. Run Schema Batch (Adapter: "schema")
        if schema_prompts:
            # print(f"Processing {len(schema_prompts)} Schema queries...")
            schema_responses = self.llm_extractor_client.call_batch(schema_prompts, adapter_name="schema")
            
            # Map back to original indices
            for original_idx, response in zip(schema_indices, schema_responses):
                final_responses[original_idx] = response


        for sample, response in zip(samples, final_responses):
            sample.prediction = response
            sample.status = Status.SUCCESS
        return samples
  
    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        
        # Step 3: Generate (Optimized Parallel)
        generated_samples = self._generate_output(samples)

        return generated_samples