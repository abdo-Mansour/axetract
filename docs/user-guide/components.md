# Components

Axetract is built with a plug-and-play philosophy. Every stage of the pipeline is a component that can be customized or replaced.

## Preprocessor

The preprocessor is responsible for turning raw input into something the LLM can understand.

- **Cleaning**: Removal of script tags, styles, and non-essential whitespace.
- **Chunking**: Dividing the DOM into semantic groups to avoid exceeding context limits.

```python
from axetract.preprocessor import AXEPreprocessor

preprocessor = AXEPreprocessor(use_clean_chunker=True)
```

## Pruner

The Pruner is a "relevance filter". It uses a 0.6B model to score HTML chunks and keep only the ones that match the user's intent.

- **Token Savings**: Often reduces the input size by 80-90%.
- **Context Preservation**: Ensures that even large pages fit into small context windows.

## Extractor

The Extractor performs the final structured mapping. It is trained to:
1. Understand the schema or natural language query.
2. Search through the pruned HTML.
3. Generate valid JSON.
4. Provide XPaths for every extracted value.

## Postprocessor

The Postprocessor ensures the output is production-ready.

- **Schema Validation**: Forcing the output to match your Pydantic model.
- **JSON Repair**: Using `json-repair` to fix common LLM formatting errors (missing braces, trailing commas).
- **Coordinate Mapping**: Merging XPaths with the JSON object.
