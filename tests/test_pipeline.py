"""Unit tests for AXEPipeline."""

from unittest.mock import MagicMock

from axetract.data_types import AXEChunk, AXEResult, AXESample, Status
from axetract.pipeline import AXEPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_components(prediction="answer", current_html="<p>html</p>"):
    """Return (preprocessor, pruner, extractor, postprocessor) mocks.

    that pass through and mutate samples realistically.
    """

    def _preprocessor_side_effect(batch):
        for s in batch:
            s.chunks = [AXEChunk(chunkid=f"{s.id}-1", content=s.content)]
            s.original_html = s.content
            s.current_html = s.content
        return batch

    def _pruner_side_effect(batch):
        for s in batch:
            s.current_html = current_html
        return batch

    def _extractor_side_effect(batch):
        for s in batch:
            s.prediction = prediction
            s.status = Status.SUCCESS
        return batch

    def _postprocessor_side_effect(batch):
        for s in batch:
            if isinstance(s.prediction, str):
                s.prediction = {"result": s.prediction}
        return batch

    preprocessor = MagicMock(side_effect=_preprocessor_side_effect)
    pruner = MagicMock(side_effect=_pruner_side_effect)
    extractor = MagicMock(side_effect=_extractor_side_effect)
    postprocessor = MagicMock(side_effect=_postprocessor_side_effect)

    return preprocessor, pruner, extractor, postprocessor


class TestAXEPipelineInit:
    def test_stores_components(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        assert pipeline.preprocessor is p
        assert pipeline.pruner is pr
        assert pipeline.extractor is e
        assert pipeline.postprocessor is pp


class TestAXEPipelineProcess:
    def test_returns_axe_result(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        result = pipeline.process("<p>Hello</p>", query="What?")
        assert isinstance(result, AXEResult)

    def test_result_has_id(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        result = pipeline.process("<p>Hello</p>", query="What?")
        assert result.id is not None
        assert len(result.id) > 0

    def test_url_content_detected(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        pipeline.process("https://example.com", query="title?")
        # preprocessor receives a sample with is_content_url=True
        call_args = p.call_args[0][0]
        assert call_args[0].is_content_url is True

    def test_inline_html_not_url(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        pipeline.process("<p>inline</p>", query="q?")
        call_args = p.call_args[0][0]
        assert call_args[0].is_content_url is False

    def test_success_result_has_no_error(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        result = pipeline.process("<p>Hello</p>", query="What?")
        assert result.status == Status.SUCCESS
        assert result.error is None

    def test_failed_result_has_error_message(self):
        def _extractor_fail(batch):
            for s in batch:
                s.status = Status.FAILED
                s.prediction = None
            return batch

        p, pr, _, pp = _make_mock_components()
        extractor = MagicMock(side_effect=_extractor_fail)

        def _pp_passthrough(batch):
            return batch

        postprocessor = MagicMock(side_effect=_pp_passthrough)

        pipeline = AXEPipeline(
            preprocessor=p, pruner=pr, extractor=extractor, postprocessor=postprocessor
        )
        result = pipeline.process("<p>Hello</p>", query="What?")
        assert result.status == Status.FAILED
        assert result.error is not None

    def test_components_called_in_order(self):
        order = []

        def _p(batch):
            order.append("preprocessor")
            for s in batch:
                s.chunks = [AXEChunk(chunkid=f"{s.id}-1", content=s.content)]
                s.current_html = s.content
            return batch

        def _pr(batch):
            order.append("pruner")
            return batch

        def _e(batch):
            order.append("extractor")
            for s in batch:
                s.status = Status.SUCCESS
                s.prediction = "done"
            return batch

        def _pp(batch):
            order.append("postprocessor")
            return batch

        pipeline = AXEPipeline(
            preprocessor=MagicMock(side_effect=_p),
            pruner=MagicMock(side_effect=_pr),
            extractor=MagicMock(side_effect=_e),
            postprocessor=MagicMock(side_effect=_pp),
        )
        pipeline.process("<p>Hi</p>", query="q?")
        assert order == ["preprocessor", "pruner", "extractor", "postprocessor"]


class TestAXEPipelineProcessMany:
    def test_returns_list_of_results(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        inputs = ["<p>One</p>", "<p>Two</p>", "<p>Three</p>"]
        results = pipeline.process_many(inputs, query="What?")
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_result_is_axe_result(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        results = pipeline.process_many(["<p>A</p>", "<p>B</p>"], query="q?")
        for r in results:
            assert isinstance(r, AXEResult)

    def test_same_query_applied_to_all(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        results = pipeline.process_many(["<p>A</p>", "<p>B</p>"], query="shared?")
        # All should succeed
        for r in results:
            assert r.status == Status.SUCCESS


class TestAXEPipelineProcessBatch:
    def test_accepts_axe_samples(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        samples = [
            AXESample(id="1", content="<p>A</p>", is_content_url=False, query="q?"),
            AXESample(id="2", content="<p>B</p>", is_content_url=False, query="q?"),
        ]
        results = pipeline.process_batch(samples)
        assert len(results) == 2

    def test_accepts_dicts(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        dicts = [
            {"input_data": "<p>A</p>", "query": "What?"},
            {"input_data": "<p>B</p>", "query": "What?"},
        ]
        results = pipeline.process_batch(dicts)
        assert len(results) == 2

    def test_dict_with_id_preserved(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        dicts = [{"id": "my-id", "input_data": "<p>X</p>", "query": "q?"}]
        results = pipeline.process_batch(dicts)
        # The result id should match the provided id
        assert results[0].id == "my-id"

    def test_mixed_dicts_and_samples(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        batch = [
            AXESample(id="s1", content="<p>A</p>", is_content_url=False, query="q?"),
            {"input_data": "<p>B</p>", "query": "q?"},
        ]
        results = pipeline.process_batch(batch)
        assert len(results) == 2

    def test_empty_batch_returns_empty(self):
        p, pr, e, pp = _make_mock_components()
        pipeline = AXEPipeline(preprocessor=p, pruner=pr, extractor=e, postprocessor=pp)
        # Preprocessor must handle empty list
        p.side_effect = lambda batch: batch
        results = pipeline.process_batch([])
        assert results == []
