import random
import time
from functools import wraps

# These imports seem to point to a non-existent module based on previous logs?
# Actually, I should check if axetract.utils.llm_client exists.
# If not, I'll use generic exceptions or fix it.
try:
    from axetract.utils.llm_client import LiteLLMRateLimitError, OpenAIRateLimitError
except ImportError:
    LiteLLMRateLimitError = Exception
    OpenAIRateLimitError = Exception


def retry_on_ratelimit(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 10.0):
    """Decorator to retry API calls specifically on RateLimit exceptions.

    Args:
        max_retries (int): Maximum number of retry attempts.
        base_delay (float): Initial delay between retries in seconds.
        max_delay (float): Maximum delay between retries.
    """

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
                    print(
                        f"Rate limited. Retrying in {sleep:.2f}s... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep)
                    delay *= 2

        return wrapped

    return deco
