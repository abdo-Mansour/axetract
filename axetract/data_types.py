from pydantic import BaseModel
from typing import List, Optional, Any, Union
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"

class AXESample(BaseModel):
    id: str
    content: str 
    is_content_url: bool
    query_or_schema: bool
    
    original_html: str = "" 
    current_html: str = "" 
    prediction: Optional[Any] = None 
    xpaths: Optional[dict] = None
    
    status: Status = Status.PENDING

class AXEResult(BaseModel):
    id: str
    prediction: Any
    xpaths: dict
    status: Status
    error: Optional[str] = None