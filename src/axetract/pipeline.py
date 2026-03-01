from __future__ import annotations
import uuid
from typing import List, Union, Optional, Type, Any
from pydantic import BaseModel

from axetract.preprocessor.base_preprocessor import BasePreprocessor
from axetract.pruner.base_pruner import BasePruner
from axetract.extractor.base_extractor import BaseExtractor
from axetract.postprocessor.base_postprocessor import BasePostprocessor

from axetract.data_types import AXESample, AXEResult, Status

class AXEPipeline:
    def __init__(self, 
                preprocessor: BasePreprocessor, 
                pruner: BasePruner, 
                extractor: BaseExtractor, 
                postprocessor: BasePostprocessor
                ):
        self.preprocessor = preprocessor
        self.pruner = pruner
        self.extractor = extractor
        self.postprocessor = postprocessor

    def process(
        self, 
        input_data: str, 
        query: Optional[str] = None, 
        schema: Optional[Type[BaseModel]] = None
    ) -> AXEResult:
        
        # 1. Create sample
        sample = AXESample(
            id=0,
            content=input_data,
            # TODO: might need to do actual util function to check if it is url
            is_content_url=input_data.strip().startswith(('http://', 'https://')),
            query=query,
            schema_model=schema
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
                error=None if sample.status == Status.SUCCESS else f"Encountered error/pending status: {sample.status}"
            ) for sample in batch
        ]
        
        return results


    