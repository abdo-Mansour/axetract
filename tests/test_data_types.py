"""Unit tests for axetract data types (AXESample, AXEChunk, AXEResult, Status)."""

import pytest
from pydantic import BaseModel, ValidationError

from axetract.data_types import AXEChunk, AXEResult, AXESample, Status


class DummySchema(BaseModel):
    name: str
    age: int


# ===========================================================================
# Status
# ===========================================================================


class TestStatus:
    def test_all_values_exist(self):
        assert Status.PENDING.value == "pending"
        assert Status.SUCCESS.value == "success"
        assert Status.FAILED.value == "failed"

    def test_comparison(self):
        assert Status.PENDING != Status.SUCCESS
        assert Status.SUCCESS == Status.SUCCESS


# ===========================================================================
# AXEChunk
# ===========================================================================


class TestAXEChunk:
    def test_creation(self):
        chunk = AXEChunk(chunkid="1-1", content="<p>Test Content</p>")
        assert chunk.chunkid == "1-1"
        assert chunk.content == "<p>Test Content</p>"

    def test_requires_chunkid_and_content(self):
        with pytest.raises(ValidationError):
            AXEChunk(chunkid="x")  # missing content

    def test_empty_content_allowed(self):
        chunk = AXEChunk(chunkid="1-1", content="")
        assert chunk.content == ""


# ===========================================================================
# AXESample
# ===========================================================================


class TestAXESample:
    def test_basic_creation(self):
        sample = AXESample(
            id="sample-1",
            content="<html><body>Hello</body></html>",
            is_content_url=False,
        )
        assert sample.id == "sample-1"
        assert sample.content == "<html><body>Hello</body></html>"
        assert sample.is_content_url is False
        assert sample.query is None
        assert sample.schema_model is None
        assert sample.status == Status.PENDING
        assert sample.chunks == []
        assert sample.original_html == ""
        assert sample.current_html == ""

    def test_with_string_query(self):
        sample = AXESample(
            id="sample-2",
            content="https://example.com",
            is_content_url=True,
            query="What is the title?",
        )
        assert sample.query == "What is the title?"

    def test_with_pydantic_schema(self):
        sample = AXESample(
            id="sample-3",
            content="Data",
            is_content_url=False,
            schema_model=DummySchema,
        )
        assert sample.schema_model == DummySchema

    def test_with_dict_schema(self):
        dict_schema = {"title": "string", "price": "number"}
        sample = AXESample(
            id="sample-4",
            content="Data",
            is_content_url=False,
            schema_model=dict_schema,
        )
        assert sample.schema_model == dict_schema

    def test_with_string_schema(self):
        sample = AXESample(
            id="sample-5",
            content="Data",
            is_content_url=False,
            schema_model='{"price": "string"}',
        )
        assert isinstance(sample.schema_model, str)

    def test_status_transition(self):
        sample = AXESample(id="s", content="x", is_content_url=False)
        assert sample.status == Status.PENDING
        sample.status = Status.SUCCESS
        assert sample.status == Status.SUCCESS
        sample.status = Status.FAILED
        assert sample.status == Status.FAILED

    def test_with_chunks(self):
        chunks = [
            AXEChunk(chunkid="1-1", content="<p>First</p>"),
            AXEChunk(chunkid="1-2", content="<p>Second</p>"),
        ]
        sample = AXESample(
            id="sample-6",
            content="<p>First</p><p>Second</p>",
            is_content_url=False,
            chunks=chunks,
        )
        assert len(sample.chunks) == 2
        assert sample.chunks[0].chunkid == "1-1"

    def test_prediction_can_be_dict(self):
        sample = AXESample(id="s", content="x", is_content_url=False)
        sample.prediction = {"price": "$10", "weight": "1kg"}
        assert sample.prediction["price"] == "$10"

    def test_xpaths_field(self):
        sample = AXESample(id="s", content="x", is_content_url=False)
        sample.xpaths = {"price": "/html/body/p[1]"}
        assert sample.xpaths["price"] == "/html/body/p[1]"


# ===========================================================================
# AXEResult
# ===========================================================================


class TestAXEResult:
    def test_basic_creation(self):
        result = AXEResult(
            id="r1",
            prediction={"price": "$10"},
            status=Status.SUCCESS,
        )
        assert result.id == "r1"
        assert result.prediction == {"price": "$10"}
        assert result.status == Status.SUCCESS
        assert result.xpaths is None
        assert result.error is None

    def test_failed_result_with_error(self):
        result = AXEResult(
            id="r2",
            prediction={},
            status=Status.FAILED,
            error="Something went wrong",
        )
        assert result.status == Status.FAILED
        assert result.error == "Something went wrong"

    def test_with_xpaths(self):
        result = AXEResult(
            id="r3",
            prediction={"price": "$10"},
            xpaths={"price": "/html/body/p"},
            status=Status.SUCCESS,
        )
        assert result.xpaths == {"price": "/html/body/p"}

    def test_string_prediction(self):
        result = AXEResult(
            id="r4",
            prediction="The price is $10",
            status=Status.SUCCESS,
        )
        assert result.prediction == "The price is $10"

    def test_requires_id_and_prediction_and_status(self):
        with pytest.raises(ValidationError):
            AXEResult(id="x", prediction={})  # missing status
