"""Shared utilities for LLM clients."""


def format_prompt_with_thinking(prompt: str, enable_thinking: bool, call_thinking: bool) -> str:
    """Format prompts for models requiring specific thinking tags.

    Args:
        prompt (str): The raw prompt text.
        enable_thinking (bool): Global thinking flag from client config.
        call_thinking (bool): Per-call thinking flag.

    Returns:
        str: Formatted prompt with appropriate thinking tags.
    """
    if enable_thinking or call_thinking:
        return f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
