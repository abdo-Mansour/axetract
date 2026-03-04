from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel


class Status(Enum):
    """Execution status for a processing sample."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class AXEChunk(BaseModel):
    """A single chunk of HTML content.

    Attributes:
        chunkid (str): Unique identifier for the chunk.
        content (str): The raw or cleaned HTML content.
    """

    chunkid: str
    content: str


class AXESample(BaseModel):
    """A data container for a single extraction request throughout the pipeline.

    Attributes:
        id (str): Unique identifier for the sample.
        content (str): Input content (URL or raw HTML).
        is_content_url (bool): Whether the content is a URL.
        query (Optional[str]): Natural language extraction query.
        schema_model (Optional[Union[str, Type[BaseModel], dict]]): Desired JSON schema.
        chunks (List[AXEChunk]): List of processed HTML chunks.
        original_html (str): The original, uncleaned HTML content.
        current_html (str): The current state of HTML (e.g., after cleaning or pruning).
        prediction (Optional[Union[str, dict, Any]]): The LLM's raw output or parsed JSON.
        xpaths (Optional[dict]): Map of extracted fields to their source XPaths.
        status (Status): Current processing status.
    """

    id: str
    content: str
    is_content_url: bool
    query: Optional[str] = None
    schema_model: Optional[Union[str, Type[BaseModel], dict]] = None
    chunks: List[AXEChunk] = []
    original_html: str = ""
    current_html: str = ""
    prediction: Optional[Union[str, dict, Any]] = None
    xpaths: Optional[dict] = None

    status: Status = Status.PENDING


class AXEResult(BaseModel):
    """Final extraction result returned to the user.

    Attributes:
        id (str): Sample identifier.
        prediction (Union[str, dict, Any]): The extracted structured data.
        xpaths (Optional[dict]): Reference XPaths for the extracted values.
        status (Status): Success or failure indicator.
        error (Optional[str]): Error message if processing failed.
    """

    id: str
    prediction: Union[str, dict, Any]
    xpaths: Optional[dict] = None
    status: Status
    error: Optional[str] = None
