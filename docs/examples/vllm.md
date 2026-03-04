# vLLM Serving

For production environments requiring high throughput, Axetract supports **vLLM** for serving the base model and dynamically routing LoRA requests.

## Setup

Ensure you have a GPU with sufficient VRAM and that `vllm` is installed in your environment.

## Usage

Simply pass `use_vllm=True` to `from_config`.

```python
from axetract import AXEPipeline

# This will initialize the LocalVLLMClient
# and load the base model into the vLLM engine.
pipeline = AXEPipeline.from_config(use_vllm=True)

# Inference remains the same, but benefits from vLLM's 
# continuous batching and efficient LoRA swapping.
result = pipeline.process(
    "https://example.com",
    query="Extract the main headline"
)
```

## Configuration

You can customize the vLLM engine by passing a config dictionary:

```python
config = {
    "model_name": "Qwen/Qwen3-0.6B",
    "engine_args": {
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
        "enable_lora": True,
        "max_loras": 4,
    }
}

pipeline = AXEPipeline.from_config(llm_config=config, use_vllm=True)
```
