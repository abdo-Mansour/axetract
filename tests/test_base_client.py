"""Unit tests for BaseClient.call_batch."""
import pytest
from unittest.mock import MagicMock, patch, call
from typing import Optional
from axetract.llm.base_client import BaseClient


# ---------------------------------------------------------------------------
# Concrete stub so we can instantiate the abstract BaseClient
# ---------------------------------------------------------------------------

class _StubClient(BaseClient):
    def call_api(self, prompt: str, adapter_name: Optional[str] = None, **kwargs) -> str:
        return f"response:{prompt}"


class _FailingClient(BaseClient):
    def __init__(self, fail_on: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._fail_on = fail_on
        self._call_count = 0

    def call_api(self, prompt: str, adapter_name: Optional[str] = None, **kwargs) -> str:
        idx = self._call_count
        self._call_count += 1
        if idx == self._fail_on:
            raise RuntimeError(f"Simulated failure on call {idx}")
        return f"ok:{prompt}"


# ===========================================================================
# Basic call_batch behaviour
# ===========================================================================

class TestCallBatch:

    def test_returns_list_same_length(self):
        client = _StubClient()
        prompts = ["a", "b", "c"]
        results = client.call_batch(prompts)
        assert len(results) == 3

    def test_results_preserve_order(self):
        client = _StubClient()
        prompts = ["x", "y", "z"]
        results = client.call_batch(prompts)
        # Each result should match expected response for its prompt
        for p, r in zip(prompts, results):
            assert r == f"response:{p}"

    def test_empty_prompts_returns_empty_list(self):
        client = _StubClient()
        assert client.call_batch([]) == []

    def test_single_prompt(self):
        client = _StubClient()
        results = client.call_batch(["only"])
        assert len(results) == 1
        assert results[0] == "response:only"

    def test_iterable_input_is_accepted(self):
        client = _StubClient()
        results = client.call_batch(iter(["a", "b"]))
        assert len(results) == 2


# ===========================================================================
# chunk_size splitting
# ===========================================================================

class TestCallBatchChunkSize:

    def test_chunk_size_splits_correctly(self):
        """With chunk_size=2 and 5 prompts, should process all 5."""
        client = _StubClient()
        prompts = [f"p{i}" for i in range(5)]
        results = client.call_batch(prompts, chunk_size=2)
        assert len(results) == 5

    def test_chunk_size_zero_processes_all(self):
        """chunk_size <= 0 means no chunking (process all at once)."""
        client = _StubClient()
        prompts = ["a", "b", "c", "d"]
        results = client.call_batch(prompts, chunk_size=0)
        assert len(results) == 4

    def test_chunk_size_larger_than_batch(self):
        """chunk_size larger than the number of prompts — should still work."""
        client = _StubClient()
        prompts = ["x", "y"]
        results = client.call_batch(prompts, chunk_size=100)
        assert len(results) == 2


# ===========================================================================
# Error handling
# ===========================================================================

class TestCallBatchErrorHandling:

    def test_failure_returns_none_by_default(self):
        """When one call fails and raise_on_error=False, result is None for that slot."""
        client = _FailingClient(fail_on=1)
        results = client.call_batch(["ok", "fail", "ok2"], raise_on_error=False)
        assert results[0] is not None   # "ok:ok"
        assert results[1] is None       # failed call → None
        assert results[2] is not None   # "ok:ok2"

    def test_failure_raises_when_requested(self):
        client = _FailingClient(fail_on=0)
        with pytest.raises(RuntimeError, match="Simulated failure"):
            client.call_batch(["will_fail"], raise_on_error=True)

    def test_all_failures_return_all_none(self):
        """Every call fails → every result is None."""
        class _AllFail(BaseClient):
            def call_api(self, prompt, adapter_name=None, **kwargs):
                raise ValueError("always fail")

        client = _AllFail()
        results = client.call_batch(["a", "b", "c"], raise_on_error=False)
        assert all(r is None for r in results)


# ===========================================================================
# per_result_callback
# ===========================================================================

class TestPerResultCallback:

    def test_callback_called_for_each_result(self):
        client = _StubClient()
        cb = MagicMock()
        prompts = ["a", "b", "c"]
        client.call_batch(prompts, per_result_callback=cb)
        assert cb.call_count == 3

    def test_callback_receives_index_result_none_error(self):
        client = _StubClient()
        received = []

        def cb(idx, res, err):
            received.append((idx, res, err))

        client.call_batch(["only"], per_result_callback=cb)
        assert len(received) == 1
        idx, res, err = received[0]
        assert idx == 0
        assert res == "response:only"
        assert err is None

    def test_callback_called_with_error_on_failure(self):
        client = _FailingClient(fail_on=0)
        received = []

        def cb(idx, res, err):
            received.append((idx, res, err))

        client.call_batch(["fail"], raise_on_error=False, per_result_callback=cb)
        assert len(received) == 1
        idx, res, err = received[0]
        assert res is None
        assert isinstance(err, RuntimeError)


# ===========================================================================
# adapter_name forwarding
# ===========================================================================

class TestAdapterName:

    def test_adapter_name_forwarded_to_call_api(self):
        """call_batch should pass adapter_name down to call_api."""
        mock_call_api = MagicMock(return_value="resp")
        client = _StubClient()
        client.call_api = mock_call_api

        client.call_batch(["prompt"], adapter_name="qa")

        mock_call_api.assert_called_once_with("prompt", adapter_name="qa")
