"""AXEtract FastAPI server exposed as the ``axe-server`` CLI entry point.

Run with::

    axe-server

or directly::

    python -m axetract.server

Environment variables
---------------------
AXE_USE_VLLM : str
    Set to ``"true"`` to use vLLM as the LLM backend. Default: ``"false"``.
AXE_PORT : int
    Port to listen on. Default: ``8000``.
AXE_HOST : str
    Host to bind to. Default: ``"0.0.0.0"``.
AXE_LOG_FILE : str
    Optional path to a log file. When unset, logs go to stderr only.
"""

import logging
import os

from dotenv import load_dotenv

# Load .env from the current working directory (or wherever the user invokes the CLI)
load_dotenv()

from axetract.utils.logging_util import setup_logging  # noqa: E402

_log_file = os.getenv("AXE_LOG_FILE")
setup_logging(log_file=_log_file if _log_file else None)

# Must be set before any vLLM import
os.environ.setdefault("VLLM_USE_V1", "0")

from typing import Any, Dict, List, Optional, Union  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from axetract.data_types import AXEResult  # noqa: E402
from axetract.pipeline import AXEPipeline  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="AXEtract API Server")

# Global pipeline instance (initialised at startup)
_pipeline: Optional[AXEPipeline] = None


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
    global _pipeline
    logger.debug("Initializing AXEPipeline...")
    use_vllm = os.getenv("AXE_USE_VLLM", "false").lower() == "true"
    try:
        _pipeline = AXEPipeline.from_config(use_vllm=use_vllm)
        logger.debug("AXEPipeline initialized successfully.")
    except Exception as exc:
        logger.error("Error initializing pipeline: %s", exc)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "pipeline_initialized": _pipeline is not None}


@app.post("/process", response_model=AXEResult)
async def process(request: ProcessRequest):
    """Extract structured data from a single input document.

    Args:
        request (ProcessRequest): The input task parameters.

    Returns:
        AXEResult: The processing result.

    Raises:
        HTTPException: 503 if pipeline not initialised; 500 on extraction error.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    logger.debug("Received process request for input: %s", request.input_data[:100] + "...")
    try:
        result = _pipeline.extract(
            input_data=request.input_data, query=request.query, schema=request.schema_model
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/process_batch", response_model=List[AXEResult])
async def process_batch(request: BatchProcessRequest):
    """Process multiple documents in a single batch.

    Args:
        request (BatchProcessRequest): List of input task parameters.

    Returns:
        List[AXEResult]: Results for each input document.

    Raises:
        HTTPException: 503 if pipeline not initialised; 500 on extraction error.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    logger.debug("Received process_batch request with %d items", len(request.items))
    try:
        batch = [
            {"input_data": item.input_data, "query": item.query, "schema": item.schema_model}
            for item in request.items
        ]
        results = _pipeline.extract_batch(batch)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main() -> None:
    """Run the AXEtract API server (entry point for ``axe-server`` CLI).

    Reads configuration from environment variables (see module docstring).
    """
    port = int(os.getenv("AXE_PORT", "8000"))
    host = os.getenv("AXE_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    main()
