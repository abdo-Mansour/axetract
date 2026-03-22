import logging
import os
import sys
from dotenv import load_dotenv
# 1. Load env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Add src to path before any axetract imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from axetract.utils.logging_util import setup_logging

setup_logging(
    log_file=os.path.join(os.path.dirname(__file__), "axe_server.log"),
)


# 2. FORCE V1 OFF (Must be done before axetract/vllm imports)
os.environ["VLLM_USE_V1"] = "0"

from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from axetract.data_types import AXEResult
from axetract.pipeline import AXEPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="AXEtract API Server")

# Global pipeline instance
pipeline = None


class ProcessRequest(BaseModel):
    """Request model for a single processing task."""

    input_data: str
    query: Optional[str] = None
    schema_model: Optional[Union[str, Dict[str, Any]]] = None


class BatchProcessRequest(BaseModel):
    """Request model for a batch of processing tasks."""

    items: List[ProcessRequest]


@app.on_event("startup")
async def startup_event():
    """Initialize the global AXEPipeline on server startup."""
    global pipeline
    logger.debug("Initializing AXEPipeline...")
    # In a real scenario, you might want to pass configs via environment variables
    use_vllm = os.getenv("AXE_USE_VLLM", "False").lower() == "true"
    try:
        pipeline = AXEPipeline.from_config(use_vllm=use_vllm)
        logger.debug("AXEPipeline initialized successfully.")
    except Exception as e:
        logger.error("Error initializing pipeline: %s", e)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "pipeline_initialized": pipeline is not None}


@app.post("/process", response_model=AXEResult)
async def process(request: ProcessRequest):
    """Extract structured data from a single input document.

    Args:
        request (ProcessRequest): The input task parameters.

    Returns:
        AXEResult: The processing result.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    logger.debug("Received process request for input: %s", request.input_data[:100] + "...")
    try:
        result = pipeline.process(
            input_data=request.input_data, query=request.query, schema=request.schema_model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_batch", response_model=List[AXEResult])
async def process_batch(request: BatchProcessRequest):
    """Process multiple documents in a single batch.

    Args:
        request (BatchProcessRequest): List of input task parameters.

    Returns:
        List[AXEResult]: Results for each input document.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    logger.debug("Received process_batch request with %d items", len(request.items))
    try:
        batch = [
            {"input_data": item.input_data, "query": item.query, "schema": item.schema_model}
            for item in request.items
        ]
        results = pipeline.process_batch(batch)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the server using uvicorn."""
    port = int(os.getenv("AXE_PORT", 8000))
    host = os.getenv("AXE_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    main()
