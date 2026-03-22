from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Optional

from axetract.llm.base_client import BaseClient
from axetract.llm.llm_utils import format_prompt_with_thinking

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = Any
    SamplingParams = Any
    LoRARequest = Any

_vllm_init_lock = threading.Lock()

class LocalVLLMClient(BaseClient):
    """Connects to a local vLLM engine for high-performance inference.

    Supports native LoRA switching and batch generation.
    """

    def __init__(self, config: dict):
        """Initialize the vLLM client.

        Args:
            config (dict): Configuration containing model_name, engine_args, etc.

        Raises:
            ImportError: If vLLM is not installed.
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with 'pip install vllm'")
        super().__init__(config)

        model_name = config.get("model_name")
        self.engine_args = config.get("engine_args", {})
        self.lora_config_raw = config.get("lora_modules", {}) or {}

        if self.lora_config_raw or config.get("enable_lora", False):
            self.engine_args["enable_lora"] = True
            self.engine_args["max_loras"] = min(len(self.lora_config_raw), 3)

        with _vllm_init_lock:
            self.llm = LLM(model=model_name, **self.engine_args)

        self.context_window_size = self.engine_args.get("max_model_len", 2048)

        self.lora_requests: Dict[str, LoRARequest] = {}
        self.adapter_defaults: Dict[str, Dict[str, Any]] = {}
        self._lora_id_counter = 1

        for name, data in self.lora_config_raw.items():
            path = data if isinstance(data, str) else data.get("path")
            defaults = (
                {} if isinstance(data, str) else {k: v for k, v in data.items() if k != "path"}
            )

            self.lora_requests[name] = LoRARequest(
                lora_name=name, lora_int_id=self._lora_id_counter, lora_path=path
            )
            self.adapter_defaults[name] = defaults
            self._lora_id_counter += 1

        self._generate_lock = threading.Lock()

        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0.0)
        self.top_p = gen_conf.get("top_p", 1.0)
        self.max_tokens = config.get("max_tokens", gen_conf.get("max_tokens", 512))
        self.stop_sequences = gen_conf.get("stop", [])
        self.enable_thinking = config.get("enable_thinking", False)

    def _create_sampling_params(self, adapter_name: str = None, **kwargs) -> SamplingParams:
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop_sequences,
        }
        if adapter_name and adapter_name in self.adapter_defaults:
            params.update(self.adapter_defaults[adapter_name])
        params.update({k: v for k, v in kwargs.items() if v is not None})

        safe_truncate_len = max(
            self.context_window_size - params["max_tokens"] - 1, self.context_window_size // 2
        )
        return SamplingParams(
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            stop=params["stop"],
            # truncate_prompt_tokens=safe_truncate_len,
        )

    def call_batch(
        self, prompts: Iterable[str], adapter_name: str = None, thinking: bool = False, **kwargs
    ) -> List[Optional[str]]:
        """High-throughput batch generation using the vLLM engine.

        Args:
            prompts (Iterable[str]): Batch of prompts.
            adapter_name (str, optional): LoRA adapter name.
            thinking (bool): Enable thinking tags.
            **kwargs: Generation parameter overrides.

        Returns:
            List[Optional[str]]: Generated texts.
        """
        sampling_params = self._create_sampling_params(adapter_name=adapter_name, **kwargs)
        prompts = [format_prompt_with_thinking(p, self.enable_thinking, thinking) for p in prompts]
        lora_req = self.lora_requests.get(adapter_name) if adapter_name else None

        with self._generate_lock:  # Protect engine state
            outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_req)
        return [out.outputs[0].text for out in outputs]

    def call_api(self, prompt: str, adapter_name: str = None, thinking=False, **kwargs) -> str:
        """Call a single prompt completion.

        Args:
            prompt (str): Input text.
            adapter_name (str, optional): LoRA adapter name.
            thinking (bool): Enable thinking tags.
            **kwargs: Generation overrides.

        Returns:
            str: Generated text.
        """
        return self.call_batch([prompt], adapter_name, thinking, **kwargs)[0]
