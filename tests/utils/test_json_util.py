"""Unit tests for json_util.is_schema and json_util.extract_and_repair_json."""

from pydantic import BaseModel

from axetract.utils.json_util import extract_and_repair_json, is_schema


class DummyModel(BaseModel):
    id: int
    name: str


# ===========================================================================
# is_schema
# ===========================================================================


class TestIsSchema:
    def test_dict_input(self):
        assert is_schema({"name": "string"}) is True

    def test_pydantic_class(self):
        assert is_schema(DummyModel) is True

    def test_json_string(self):
        assert is_schema('{"name": "string"}') is True

    def test_dict_keys_string(self):
        assert is_schema("dict_keys(['name', 'age'])") is True

    def test_plain_question_is_not_schema(self):
        assert is_schema("What is the title?") is False

    def test_none_is_not_schema(self):
        assert is_schema(None) is False

    def test_integer_is_not_schema(self):
        assert is_schema(42) is False

    def test_empty_string_is_not_schema(self):
        assert is_schema("") is False

    def test_json_array_string(self):
        assert is_schema('["a", "b"]') is True

    def test_colon_rich_string_treated_as_schema(self):
        # Two or more colons triggers the heuristic
        assert is_schema("price: string, weight: string, count: int") is True


# ===========================================================================
# extract_and_repair_json
# ===========================================================================


class TestExtractAndRepairJson:
    def test_valid_json(self):
        result = extract_and_repair_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced_json(self):
        md = '```json\n{"price": 100}\n```'
        result = extract_and_repair_json(md)
        assert result == {"price": 100}

    def test_malformed_json_repaired(self):
        malformed = '{name: "test"'
        result = extract_and_repair_json(malformed)
        assert isinstance(result, dict)
        assert result.get("name") == "test"

    def test_none_returns_empty_dict(self):
        assert extract_and_repair_json(None) == {}

    def test_dict_input_returned_as_is(self):
        d = {"already": "dict"}
        assert extract_and_repair_json(d) == d

    def test_spread_values_concatenates_strings(self):
        result = extract_and_repair_json('{"k1": "Hello", "k2": " World"}', spread_values=True)
        assert isinstance(result, str)
        assert "Hello" in result
        assert "World" in result

    def test_spread_values_skips_non_strings(self):
        result = extract_and_repair_json('{"k1": "Hello", "k2": 123}', spread_values=True)
        assert "Hello" in result
        # 123 should be skipped (not a string)

    def test_embedded_json_block_extracted(self):
        response = 'Some prefix text {"a": 1} some suffix'
        result = extract_and_repair_json(response)
        assert result.get("a") == 1
