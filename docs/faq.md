# FAQ & Troubleshooting

## General

### What models does Axetract support?
Axetract is optimized for the **Qwen** family of models, specifically **Qwen3-0.6B**. However, the `BaseLLMClient` can be extended to support any `transformers` or `vLLM` compatible model.

### Can I run this without a GPU?
Yes, using the `HuggingFaceClient`. However, inference with even small models like 0.6B will be significantly slower on a CPU.

## Common Issues

### "CUDA Out of Memory"
If you are using `vLLM`, try reducing `gpu_memory_utilization` in the config:
```python
config = {"engine_args": {"gpu_memory_utilization": 0.5}}
pipeline = AXEPipeline.from_config(llm_config=config, use_vllm=True)
```

### Path resolution errors
Ensure you are running the server or scripts from the root of the project, or that the `src` directory is in your `PYTHONPATH`.

### Low extraction accuracy
- Ensure your `query` is specific.
- If using a `schema`, ensure the field names are descriptive.
- Large documents might require a higher `max_tokens` setting.

## LoRA Issues

### Adapter not found
Ensure you have an active internet connection to download the adapters from Hugging Face on the first run. If you are in an air-gapped environment, pre-download the models and point to the local paths in the `llm_config`.
