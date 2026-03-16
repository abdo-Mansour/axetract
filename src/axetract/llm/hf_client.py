import logging
import threading
from typing import Iterable, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from axetract.llm.base_client import BaseClient
from axetract.llm.llm_utils import format_prompt_with_thinking

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseClient):
    """Connects directly to Hugging Face transformers.

    Supports native tensor batching and on-the-fly PEFT LoRA switching.
    Optimized for maximum throughput with Flash Attention 2, torch.compile,
    and static KV cache when available.

    Performance optimizations in call_batch:
    - Phase 1: All tokenization happens OUTSIDE the GPU lock (CPU work)
    - Phase 2: Dynamic batching by token length OUTSIDE the lock
    - Phase 3: Tensor padding/construction OUTSIDE the lock
    - Phase 4: Only adapter switch + model.generate inside the lock
    - Phase 5: Token decoding OUTSIDE the lock
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
        # Support max_tokens at top level or in generation_config
        self.max_tokens = config.get("max_tokens", gen_conf.get("max_tokens", 512))
        self.enable_thinking = config.get("enable_thinking", False)

        # Read max_model_len from engine_args or use a safe fallback
        engine_args = config.get("engine_args", {})
        self.max_model_len = engine_args.get("max_model_len", 8192)

    def _get_generation_config(self, adapter_name: Optional[str] = None, **kwargs) -> dict:
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_tokens,
            # Explicitly enable KV-cache: without this, every decode step
            # recomputes the full attention matrix instead of appending to cache.
            "use_cache": True,
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
        """Optimized batch generation with minimal GPU lock scope.

        All CPU work (tokenization, sorting, padding, decoding) happens
        OUTSIDE the GPU lock. Only adapter switching and model.generate()
        are protected, maximizing GPU utilization in concurrent pipelines.

        Architecture:
          Phase 1 (CPU, unlocked): Pre-tokenize all prompts (single pass)
          Phase 2 (CPU, unlocked): Dynamic batching by token length
          Phase 3 (CPU, unlocked): Pad and build tensors per chunk
          Phase 4 (GPU, LOCKED):   Adapter switch + generate for each chunk
          Phase 5 (CPU, unlocked): Decode generated tokens

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

        if not prompts:
            return []

        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Pre-tokenize ALL prompts OUTSIDE the lock (CPU)
        # Single-pass tokenization — no double encode for lengths.
        # ═══════════════════════════════════════════════════════════════
        pre_tokenized = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_model_len,
            return_attention_mask=False,
        )
        token_ids_list = pre_tokenized["input_ids"]
        token_lengths = [len(ids) for ids in token_ids_list]

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Dynamic batching by token length OUTSIDE the lock
        # Groups similar-length sequences to minimize padding waste.
        # ═══════════════════════════════════════════════════════════════
        sorted_indices = sorted(range(len(prompts)), key=lambda i: token_lengths[i])

        batches = []
        current_batch = []
        max_len_in_batch = 0

        for idx in sorted_indices:
            tlen = token_lengths[idx]
            if not current_batch:
                current_batch.append(idx)
                max_len_in_batch = tlen
            elif len(current_batch) >= chunk_size or tlen > max_len_in_batch * 1.3 + 50:
                batches.append(current_batch)
                current_batch = [idx]
                max_len_in_batch = tlen
            else:
                current_batch.append(idx)
                max_len_in_batch = max(max_len_in_batch, tlen)

        if current_batch:
            batches.append(current_batch)

        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Pad each chunk into tensors OUTSIDE the lock (CPU)
        # Left-padding for causal LM generation.
        # ═══════════════════════════════════════════════════════════════
        pad_id = self.tokenizer.pad_token_id
        prepared_batches = []

        for batch_indices in batches:
            batch_ids = [token_ids_list[i] for i in batch_indices]
            max_len = max(len(ids) for ids in batch_ids)

            padded_input_ids = []
            attention_masks = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded_input_ids.append([pad_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))

            prepared_batches.append({
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            })

        # ═══════════════════════════════════════════════════════════════
        # Phase 4: GPU inference INSIDE the lock (minimal critical section)
        # Only adapter switch + model.generate are protected.
        # ═══════════════════════════════════════════════════════════════
        raw_generated = {}  # orig_idx -> list of token ids (on CPU)
        context_mgr = None

        with self._generate_lock:
            # Switch adapter
            if adapter_name and hasattr(self.model, "set_adapter"):
                self.model.set_adapter(adapter_name)
            elif hasattr(self.model, "set_adapter"):
                context_mgr = self.model.disable_adapter()
                context_mgr.__enter__()

            try:
                for batch_indices, tensors in zip(batches, prepared_batches):
                    inputs = {k: v.to(self.model.device) for k, v in tensors.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **gen_kwargs)

                    # Slice off prompt tokens and move to CPU immediately
                    input_length = inputs["input_ids"].shape[1]
                    generated_tokens = outputs[:, input_length:].cpu()

                    for i, orig_idx in enumerate(batch_indices):
                        raw_generated[orig_idx] = generated_tokens[i].tolist()

            finally:
                if context_mgr is not None:
                    try:
                        context_mgr.__exit__(None, None, None)
                    except Exception:
                        pass

        # ═══════════════════════════════════════════════════════════════
        # Phase 5: Decode OUTSIDE the lock (CPU work)
        # Frees the GPU lock for other threads/pipeline stages sooner.
        # ═══════════════════════════════════════════════════════════════
        ordered_tokens = [raw_generated[i] for i in range(len(prompts))]
        all_results = self.tokenizer.batch_decode(ordered_tokens, skip_special_tokens=True)

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
