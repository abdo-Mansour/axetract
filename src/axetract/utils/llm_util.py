import random
import time
from functools import wraps
from axetract.utils.llm_client import LiteLLMRateLimitError, OpenAIRateLimitError

def retry_on_ratelimit(max_retries=5, base_delay=1.0, max_delay=10.0):
    """Decorator to retry API calls specifically on RateLimit exceptions."""
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except (LiteLLMRateLimitError, OpenAIRateLimitError) as e:
                    if attempt == max_retries - 1:
                        raise e
                    sleep = min(max_delay, delay) + random.uniform(0, delay)
                    print(f"Rate limited. Retrying in {sleep:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep)
                    delay *= 2
        return wrapped
    return deco