from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Any, Union, Type
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"

class AXEChunk(BaseModel):
    chunkid: str
    content: str

class AXESample(BaseModel):
    id: str
    content: str 
    is_content_url: bool
    query: Optional[str] = None
    schema_model: Optional[Type[BaseModel]] = None
    chunks: List[AXEChunk] = []
    original_html: str = "" 
    current_html: str = "" 
    prediction: Optional[Union[str, dict, Any]] = None 
    xpaths: Optional[dict] = None
    
    status: Status = Status.PENDING

class AXEResult(BaseModel):
    id: str
    prediction: Union[str, dict, Any]
    xpaths: Optional[dict] = None
    status: Status
    error: Optional[str] = None