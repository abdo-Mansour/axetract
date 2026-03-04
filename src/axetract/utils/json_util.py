import json
import re
from typing import Any, Union

from json_repair import repair_json


def is_schema(text: Any) -> bool:
    """Check if a text blob looks like a JSON schema or a dictionary key list.

    Args:
        text (Any): The input to check.

    Returns:
        bool: True if it looks like a schema.
    """
    if isinstance(text, dict):
        return True

    if not isinstance(text, str):
        # Check if it's a Pydantic BaseModel class
        try:
            from pydantic import BaseModel

            if isinstance(text, type) and issubclass(text, BaseModel):
                return True
        except (ImportError, TypeError):
            pass
        return False

    # Case 1: Python dict_keys
    if text.strip().startswith("dict_keys("):
        return True

    # Case 2: Looks like JSON or Python dict
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            json.loads(text)  # if it's valid JSON
            return True
        except Exception:
            return True  # maybe Python dict-like, not strict JSON

    # Case 3: Contains lots of colons/commas (schema-like pattern)
    if len(re.findall(r":", text)) >= 2:
        return True

    return False


def extract_and_repair_json(response: Union[str, dict], spread_values: bool = False) -> dict:
    """Worker function for processing a single LLM response.

    Returns a dict (empty dict on error). Top-level function for pickling.

    Args:
        response (Union[str, dict]): The raw LLM string or an already parsed dict.
        spread_values (bool): If True, concatenates all string values in the resulting dict.

    Returns:
        dict: The repaired and parsed JSON object.
    """
    if response is None:
        return {}

    if isinstance(response, dict):
        return response  # already a dict

    try:
        json_string = response

        # common fenced codeblock ```json ... ```
        if "```json" in response:
            try:
                json_string = response.split("```json", 1)[1].split("```", 1)[0]
            except Exception:
                json_string = response

        # fallback: try to capture the first {...} block
        if "{" in json_string and "}" in json_string:
            start_index = json_string.find("{")
            end_index = json_string.rfind("}") + 1
            json_string = json_string[start_index:end_index]

        repaired = repair_json(json_string)
        parsed = json.loads(repaired)

        # For non Schema queries
        if spread_values:
            final_parsing = ""
            for _, val in list(parsed.items()):
                if not isinstance(val, str):
                    continue
                final_parsing += val
            return final_parsing

        return parsed
    except Exception:
        return response
