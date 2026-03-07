import logging
import threading
from typing import Iterable, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from axetract.llm.base_client import BaseClient

logger = logging.getLogger(__name__)

# 3. Hugging Face Transformers & PEFT
try:
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def format_prompt_with_thinking(prompt: str, enable_thinking: bool, call_thinking: bool) -> str:
    """Helper to format prompts for models requiring specific thinking tags."""
    if enable_thinking or call_thinking:
        return f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class HuggingFaceClient(BaseClient):
    """Connects directly to Hugging Face transformers.

    Supports native tensor batching and on-the-fly PEFT LoRA switching.
    Optimized for maximum throughput with Flash Attention 2, torch.compile,
    and static KV cache when available.
    """

    def __init__(self, config: dict):
        """Initialize the Hugging Face client.

        Args:
            config (dict): Configuration containing model_name, lora_modules, etc.

        Raises:
            ImportError: If torch/transformers/peft are not installed.
            ValueError: If model_name is missing.
        """
        super().__init__(config)

        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for HuggingFaceClient")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Required for batch generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load Base Model with performance optimizations
        model_kwargs = config.get(
            "model_kwargs", {"device_map": "auto", "torch_dtype": torch.float16}
        )

        # ── Flash Attention 2 (opt-in) ──
        # Can give ~2-4x faster attention, but may conflict with PEFT/LoRA.
        # Only enable if you've tested it with your specific model+adapter setup.
        use_flash_attn = config.get("use_flash_attention", False)
        if use_flash_attn and "attn_implementation" not in model_kwargs:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.debug("Enabling Flash Attention 2 for faster inference.")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except (ValueError, ImportError) as e:
            # Flash Attention 2 unavailable — fall back to default
            if "flash" in str(e).lower() or "attn_implementation" in str(e).lower():
                logger.warning(
                    "Flash Attention 2 not available (%s). Falling back to default attention.", e
                )
                model_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                raise

        # Load LoRA Adapters using PEFT
        self.lora_config_raw = config.get("lora_modules", {}) or {}
        self.adapter_defaults = {}

        if self.lora_config_raw:
            adapters = list(self.lora_config_raw.items())
            # Initialize PeftModel with the first adapter
            first_name, first_data = adapters[0]
            first_path = first_data if isinstance(first_data, str) else first_data.get("path")
            self.model = PeftModel.from_pretrained(self.model, first_path, adapter_name=first_name)
            self.adapter_defaults[first_name] = (
                {}
                if isinstance(first_data, str)
                else {k: v for k, v in first_data.items() if k != "path"}
            )

            # Load remaining adapters
            for name, data in adapters[1:]:
                path = data if isinstance(data, str) else data.get("path")
                self.model.load_adapter(path, adapter_name=name)
                self.adapter_defaults[name] = (
                    {} if isinstance(data, str) else {k: v for k, v in data.items() if k != "path"}
                )

        self.model.eval()

        # ── torch.compile ──
        # JIT-compiles the model graph, eliminating Python overhead and fusing
        # GPU kernels. Gives ~1.5-2x speedup after warmup.
        use_compile = config.get("use_torch_compile", False)
        if use_compile:
            try:
                compile_mode = config.get("compile_mode", "reduce-overhead")
                logger.debug("Compiling model with torch.compile (mode=%s)...", compile_mode)
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.debug("torch.compile succeeded.")
            except Exception as e:
                logger.warning("torch.compile failed (%s). Continuing without compilation.", e)

        self._generate_lock = threading.Lock()  # Vital to prevent adapter state collision

        # Defaults
        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0.0)
        self.top_p = gen_conf.get("top_p", 1.0)
        self.max_tokens = gen_conf.get("max_tokens", 512)
        self.enable_thinking = config.get("enable_thinking", False)

    def _get_generation_config(self, adapter_name: Optional[str] = None, **kwargs) -> dict:
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_tokens,
        }
        if adapter_name and adapter_name in self.adapter_defaults:
            params.update(self.adapter_defaults[adapter_name])

        # Translate unified 'max_tokens' to HF's 'max_new_tokens'
        if "max_tokens" in kwargs:
            params["max_new_tokens"] = kwargs.pop("max_tokens")
        params.update({k: v for k, v in kwargs.items() if v is not None})

        # Hugging Face logic: If temp is 0, greedy decode (do_sample=False)
        params["do_sample"] = params.get("temperature", 0.0) > 0.0
        if not params["do_sample"]:
            params.pop("temperature", None)
            params.pop("top_p", None)

        params["pad_token_id"] = self.tokenizer.pad_token_id
        return params

    def call_batch(
        self,
        prompts: Iterable[str],
        adapter_name: Optional[str] = None,
        chunk_size: int = 8,
        thinking: bool = False,
        **kwargs,
    ) -> List[Optional[str]]:
        """Overrides default threaded batching with proper tensor-level Hugging Face batching.

        Uses `chunk_size` to prevent GPU Out-Of-Memory errors.

        Args:
            prompts (Iterable[str]): Batch of prompts.
            adapter_name (Optional[str]): Target LoRA adapter.
            chunk_size (int): Internal batch size for GPU processing.
            thinking (bool): Enable thinking tags in prompt.
            **kwargs: Generation parameter overrides.

        Returns:
            List[Optional[str]]: Decoded completions.
        """
        prompts = [format_prompt_with_thinking(p, self.enable_thinking, thinking) for p in prompts]
        gen_kwargs = self._get_generation_config(adapter_name=adapter_name, **kwargs)
        all_results = []

        with self._generate_lock:
            # 1. Switch Adapter
            if adapter_name and hasattr(self.model, "set_adapter"):
                self.model.set_adapter(adapter_name)
            elif hasattr(self.model, "set_adapter"):
                # Disable PEFT if falling back to base model
                context_mgr = self.model.disable_adapter()
                context_mgr.__enter__()

            try:
                # 2. Process in manageable chunks to avoid OOM
                for i in range(0, len(prompts), chunk_size):
                    chunk = prompts[i : i + chunk_size]

                    inputs = self.tokenizer(
                        chunk, return_tensors="pt", padding=True, truncation=True
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **gen_kwargs)

                    # 3. Slice the output to exclude the prompt tokens
                    input_length = inputs["input_ids"].shape[1]
                    generated_tokens = outputs[:, input_length:]

                    decoded = self.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    all_results.extend(decoded)
            finally:
                # Cleanup context manager if we disabled adapters
                if not adapter_name and hasattr(self.model, "set_adapter"):
                    # This check is a bit redundant but safe
                    try:
                        context_mgr.__exit__(None, None, None)
                    except NameError:
                        pass

        return all_results

    def call_api(
        self, prompt: str, adapter_name: Optional[str] = None, thinking=False, **kwargs
    ) -> str:
        """Call a single prompt completion.

        Args:
            prompt (str): Input text.
            adapter_name (Optional[str]): LoRA adapter name.
            thinking (bool): Enable thinking tags.
            **kwargs: Generation overrides.

        Returns:
            str: Generated text.
        """
        return self.call_batch(
            [prompt], adapter_name=adapter_name, chunk_size=1, thinking=thinking, **kwargs
        )[0]
