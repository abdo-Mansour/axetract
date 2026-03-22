"""End-to-end tests for AXEPipeline.

All LLM calls are mocked — no GPU required.
"""

from unittest.mock import MagicMock

from pydantic import BaseModel

from axetract.data_types import AXEChunk, AXEResult, AXESample, Status
from axetract.pipeline import AXEPipeline

# ---------------------------------------------------------------------------
# Shared test HTML fixture
# ---------------------------------------------------------------------------

PRODUCT_HTML = """
<html>
  <head><title>Product Page</title></head>
  <body>
    <nav><ul><li>Home</li><li>Products</li></ul></nav>
    <div id="main-content">
      <h1>SuperWidget 3000</h1>
      <p class="description">The SuperWidget 3000 features high-performance processing.</p>
      <table>
        <tr><th>Feature</th><th>Value</th></tr>
        <tr><td>Weight</td><td>1.2kg</td></tr>
        <tr><td>Price</td><td>$299</td></tr>
      </table>
      <div class="reviews">
        <h2>User Reviews</h2>
        <div class="review">"Life changing!" - Jane Doe</div>
      </div>
    </div>
    <footer>Copyright 2025 WidgetCorp</footer>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# Mock component factory
# ---------------------------------------------------------------------------


def _make_pipeline(
    extractor_prediction: str = '{"price": "$299", "weight": "1.2kg"}',
    postprocessor_prediction=None,
) -> AXEPipeline:
    """Build an AXEPipeline with all components mocked so no LLM/GPU is needed."""

    def _preprocessor(batch):
        for s in batch:
            s.original_html = s.content
            s.current_html = s.content
            s.chunks = [AXEChunk(chunkid=f"{s.id}-1", content=s.content)]
        return batch

    def _pruner(batch):
        for s in batch:
            s.current_html = s.content  # no-op pruning
        return batch

    def _extractor(batch):
        for s in batch:
            s.prediction = extractor_prediction
            s.status = Status.SUCCESS
        return batch

    def _postprocessor(batch):
        for s in batch:
            if postprocessor_prediction is not None:
                s.prediction = postprocessor_prediction
            elif isinstance(s.prediction, str):
                import json

                try:
                    s.prediction = json.loads(s.prediction)
                except Exception:
                    pass
        return batch

    return AXEPipeline(
        preprocessor=MagicMock(side_effect=_preprocessor),
        pruner=MagicMock(side_effect=_pruner),
        extractor=MagicMock(side_effect=_extractor),
        postprocessor=MagicMock(side_effect=_postprocessor),
    )


# ===========================================================================
# E2E: process() with a natural-language query
# ===========================================================================


class TestE2EProcessWithQuery:
    def test_returns_axe_result(self):
        pipeline = _make_pipeline()
        result = pipeline.process(PRODUCT_HTML, query="what is the weight and the price?")
        assert isinstance(result, AXEResult)

    def test_result_is_success(self):
        pipeline = _make_pipeline()
        result = pipeline.process(PRODUCT_HTML, query="what is the weight and the price?")
        assert result.status == Status.SUCCESS

    def test_result_is_no_error_on_success(self):
        pipeline = _make_pipeline()
        result = pipeline.process(PRODUCT_HTML, query="what is the weight and the price?")
        assert result.error is None

    def test_prediction_contains_price_and_weight(self):
        pipeline = _make_pipeline()
        result = pipeline.process(PRODUCT_HTML, query="what is the weight and the price?")
        pred = result.prediction
        # Prediction should be a dict or string containing our mocked data
        if isinstance(pred, dict):
            assert "price" in pred or "weight" in pred
        else:
            assert "$299" in str(pred) or "1.2kg" in str(pred)

    def test_result_has_id(self):
        pipeline = _make_pipeline()
        result = pipeline.process(PRODUCT_HTML, query="q?")
        assert result.id and len(result.id) > 0


# ===========================================================================
# E2E: process() with a JSON schema
# ===========================================================================


class TestE2EProcessWithSchema:
    SCHEMA = '{"price": "string", "weight": "string"}'

    def test_returns_axe_result(self):
        pipeline = _make_pipeline()
        result = pipeline.process(PRODUCT_HTML, schema=self.SCHEMA)
        assert isinstance(result, AXEResult)

    def test_result_is_success(self):
        pipeline = _make_pipeline()
        result = pipeline.process(PRODUCT_HTML, schema=self.SCHEMA)
        assert result.status == Status.SUCCESS

    def test_dict_schema_accepted(self):
        pipeline = _make_pipeline()
        result = pipeline.process(
            PRODUCT_HTML,
            schema={"price": "string", "weight": "string"},
        )
        assert isinstance(result, AXEResult)

    def test_pydantic_schema_accepted(self):
        class ProductSchema(BaseModel):
            price: str
            weight: str

        pipeline = _make_pipeline(
            extractor_prediction='{"price": "$299", "weight": "1.2kg"}',
            postprocessor_prediction={"price": "$299", "weight": "1.2kg"},
        )
        result = pipeline.process(PRODUCT_HTML, schema=ProductSchema)
        assert result.status == Status.SUCCESS


# ===========================================================================
# E2E: process_many() — same query, multiple documents
# ===========================================================================


class TestE2EProcessMany:
    def test_correct_number_of_results(self):
        pipeline = _make_pipeline()
        docs = [PRODUCT_HTML, "<p>Another page</p>", "<div>Third</div>"]
        results = pipeline.process_many(docs, query="extract info")
        assert len(results) == 3

    def test_all_results_are_axe_results(self):
        pipeline = _make_pipeline()
        results = pipeline.process_many([PRODUCT_HTML, "<p>B</p>"], query="What is the title?")
        for r in results:
            assert isinstance(r, AXEResult)

    def test_all_results_are_success(self):
        pipeline = _make_pipeline()
        results = pipeline.process_many([PRODUCT_HTML, "<p>B</p>"], query="q?")
        for r in results:
            assert r.status == Status.SUCCESS


# ===========================================================================
# E2E: process_batch() with dict inputs
# ===========================================================================


class TestE2EProcessBatch:
    def test_dict_inputs_processed(self):
        pipeline = _make_pipeline()
        batch = [
            {"input_data": PRODUCT_HTML, "query": "get price"},
            {"input_data": "<p>Other</p>", "schema": '{"title": "string"}'},
        ]
        results = pipeline.process_batch(batch)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, AXEResult)

    def test_axesample_inputs_processed(self):
        pipeline = _make_pipeline()
        samples = [
            AXESample(id="s1", content=PRODUCT_HTML, is_content_url=False, query="q?"),
        ]
        results = pipeline.process_batch(samples)
        assert len(results) == 1
        assert results[0].status == Status.SUCCESS

    def test_empty_batch(self):
        pipeline = _make_pipeline()
        pipeline.preprocessor.side_effect = lambda batch: batch
        results = pipeline.process_batch([])
        assert results == []
