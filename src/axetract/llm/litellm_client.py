import os
import random
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import List, Iterable, Optional, Any, Callable, Dict, Union
from axetract.utils.llm_util import retry_on_ratelimit
from axetract.llm.base_client import BaseClient
# 1. LiteLLM
try:
    import litellm
    from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    LiteLLMRateLimitError = Exception

# Standard OpenAI Exception Fallback
try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = LiteLLMRateLimitError




class LiteLLMClient(BaseClient):
    def __init__(self, config: dict):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is not installed. Install with 'pip install litellm'")
        super().__init__(config)
        
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.api_base = config.get("api_base") 
        self.custom_llm_provider = config.get("custom_llm_provider") 
        
        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0.0)
        self.top_p = gen_conf.get("top_p", 1.0)
        self.max_tokens = gen_conf.get("max_tokens", 8192)
        self.extra_body = config.get("extra_body", {})

    @retry_on_ratelimit(max_retries=10, base_delay=0.5, max_delay=5.0)
    def call_api(self, prompt: str, adapter_name: Optional[str] = None, **kwargs) -> str:
        # standard remote vLLM uses `model` to specify the LoRA adapter name
        target_model = adapter_name if adapter_name else self.model_name
        
        temperature = kwargs.pop("temperature", self.temperature)
        top_p = kwargs.pop("top_p", self.top_p)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        
        call_extra_body = self.extra_body.copy()
        if "extra_body" in kwargs:
            call_extra_body.update(kwargs.pop("extra_body"))

        messages = kwargs.pop("messages",[
            {"role": "system", "content": "Reasoning: high"}, 
            {"role": "user", "content": prompt}
        ])

        litellm_args = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        
        if self.api_base: litellm_args["api_base"] = self.api_base
        if self.api_key: litellm_args["api_key"] = self.api_key
        if self.custom_llm_provider: litellm_args["custom_llm_provider"] = self.custom_llm_provider
        if call_extra_body: litellm_args["extra_body"] = call_extra_body
        litellm_args.update(kwargs)

        response = litellm.completion(**litellm_args)
        return response.choices[0].message.content

