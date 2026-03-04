"""Unit tests for AXEExtractor."""
import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from axetract.extractor.axe_extractor import AXEExtractor
from axetract.data_types import AXESample, AXEChunk, Status


SCHEMA_PROMPT = "Schema: {query}\nContent: {content}"
QA_PROMPT = "Question: {query}\nContent: {content}"


def _make_sample(
    sample_id: str = "1",
    query: str = None,
    schema_model=None,
    current_html: str = "<p>Test content</p>",
) -> AXESample:
    return AXESample(
        id=sample_id,
        content=current_html,
        is_content_url=False,
        query=query,
        schema_model=schema_model,
        current_html=current_html,
    )


class TestAXEExtractorInit:
    def test_stores_config(self):
        mock_llm = MagicMock()
        extractor = AXEExtractor(
            llm_extractor_client=mock_llm,
            schema_generation_prompt_template=SCHEMA_PROMPT,
            query_generation_prompt_template=QA_PROMPT,
            batch_size=32,
            num_workers=8,
        )
        assert extractor.llm_extractor_client is mock_llm
        assert extractor.batch_size == 32
        assert extractor.num_workers == 8
        assert extractor.schema_prompt_template == SCHEMA_PROMPT
        assert extractor.query_prompt_template == QA_PROMPT

    def test_default_batch_size(self):
        mock_llm = MagicMock()
        extractor = AXEExtractor(
            llm_extractor_client=mock_llm,
            schema_generation_prompt_template=SCHEMA_PROMPT,
            query_generation_prompt_template=QA_PROMPT,
        )
        assert extractor.batch_size == 16
        assert extractor.num_workers == 4


class TestAXEExtractorGenerateOutput:
    def _make_extractor(self, mock_llm=None):
        if mock_llm is None:
            mock_llm = MagicMock()
        return AXEExtractor(
            llm_extractor_client=mock_llm,
            schema_generation_prompt_template=SCHEMA_PROMPT,
            query_generation_prompt_template=QA_PROMPT,
        ), mock_llm

    def test_qa_query_uses_qa_adapter(self):
        extractor, mock_llm = self._make_extractor()
        mock_llm.call_batch.return_value = ['{"answer": "blue"}']

        sample = _make_sample(query="What color is it?")
        results = extractor._generate_output([sample])

        mock_llm.call_batch.assert_called_once()
        call_kwargs = mock_llm.call_batch.call_args
        assert call_kwargs[1].get("adapter_name") == "qa" or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "qa"
        ) or call_kwargs.kwargs.get("adapter_name") == "qa"

    def test_schema_query_uses_schema_adapter(self):
        extractor, mock_llm = self._make_extractor()
        mock_llm.call_batch.return_value = ['{"price": "$10"}']

        sample = _make_sample(schema_model={"price": "string"})
        results = extractor._generate_output([sample])

        mock_llm.call_batch.assert_called_once()
        call_kwargs = mock_llm.call_batch.call_args
        assert call_kwargs.kwargs.get("adapter_name") == "schema"

    def test_sample_prediction_set_after_extraction(self):
        extractor, mock_llm = self._make_extractor()
        mock_llm.call_batch.return_value = ['{"result": "ok"}']

        sample = _make_sample(query="question?")
        results = extractor._generate_output([sample])

        assert results[0].prediction == '{"result": "ok"}'
        assert results[0].status == Status.SUCCESS

    def test_mixed_qa_and_schema_samples(self):
        mock_llm = MagicMock()
        mock_llm.call_batch.side_effect = [
            ['{"qa_answer": "yes"}'],    # QA call
            ['{"schema_price": "$5"}'],  # Schema call
        ]
        extractor = AXEExtractor(
            llm_extractor_client=mock_llm,
            schema_generation_prompt_template=SCHEMA_PROMPT,
            query_generation_prompt_template=QA_PROMPT,
        )

        qa_sample = _make_sample(sample_id="1", query="What is it?")
        schema_sample = _make_sample(sample_id="2", schema_model={"price": "string"})

        results = extractor._generate_output([qa_sample, schema_sample])

        assert mock_llm.call_batch.call_count == 2
        assert results[0].status == Status.SUCCESS
        assert results[1].status == Status.SUCCESS

    def test_pydantic_schema_converted_to_string_in_prompt(self):
        class MyModel(BaseModel):
            name: str
            count: int

        extractor, mock_llm = self._make_extractor()
        mock_llm.call_batch.return_value = ['{"name": "thing", "count": 3}']

        sample = _make_sample(schema_model=MyModel)
        results = extractor._generate_output([sample])

        # Should succeed - pydantic model converted internally
        assert results[0].status == Status.SUCCESS

    def test_dict_schema_converted_to_string_in_prompt(self):
        extractor, mock_llm = self._make_extractor()
        mock_llm.call_batch.return_value = ['{"price": "$10"}']

        sample = _make_sample(schema_model={"price": "string", "weight": "string"})
        results = extractor._generate_output([sample])

        assert results[0].status == Status.SUCCESS


class TestAXEExtractorCall:
    def test_call_delegates_to_generate_output(self):
        mock_llm = MagicMock()
        mock_llm.call_batch.return_value = ["answer"]
        extractor = AXEExtractor(
            llm_extractor_client=mock_llm,
            schema_generation_prompt_template=SCHEMA_PROMPT,
            query_generation_prompt_template=QA_PROMPT,
        )
        sample = _make_sample(query="q?")
        results = extractor([sample])
        assert len(results) == 1
        assert results[0].status == Status.SUCCESS

    def test_empty_batch(self):
        mock_llm = MagicMock()
        extractor = AXEExtractor(
            llm_extractor_client=mock_llm,
            schema_generation_prompt_template=SCHEMA_PROMPT,
            query_generation_prompt_template=QA_PROMPT,
        )
        results = extractor([])
        assert results == []
        mock_llm.call_batch.assert_not_called()
