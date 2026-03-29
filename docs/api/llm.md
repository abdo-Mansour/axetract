# LLM Clients API

AXEtract abstracts LLM interactions through a unified client interface, supporting HuggingFace local execution, vLLM high-throughput serving, and LiteLLM for API-based models.

## Base Client

::: axetract.llm.base_client.BaseClient
    options:
      show_root_heading: true
      show_source: true

## Hugging Face Client

::: axetract.llm.hf_client.HuggingFaceClient
    options:
      show_root_heading: true
      show_source: true

## vLLM Client

::: axetract.llm.vllm_client.LocalVLLMClient
    options:
      show_root_heading: true
      show_source: true

## LiteLLM Client

::: axetract.llm.litellm_client.LiteLLMClient
    options:
      show_root_heading: true
      show_source: true
