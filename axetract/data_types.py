from pydantic import BaseModel
from typing import List, Optional, Any, Union


class AXESample(BaseModel):
    id: str
    content: str 
    is_content_url: bool
    query_or_schema: bool
    
    original_html: str = "" 
    current_html: str = "" 
    prediction: Optional[Any] = None 
    xpaths: Optional[dict] = None
    
    status: str = "pending" # pending, success, failed
    error: Optional[str] = None