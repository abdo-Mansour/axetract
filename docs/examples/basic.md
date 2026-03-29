# Basic Extraction

This example shows how to perform a simple extraction from a URL using a natural language query.

## Code

```python
from axetract import AXEPipeline

# Initialize the pipeline with default LoRA adapters
# This will use HuggingFaceClient by default
pipeline = AXEPipeline.from_config()

# Define the URL and the query
url = "https://example.com/products/smartphone-x"
query = "What is the price, model name, and color of the device?"

# Run the pipeline
result = pipeline.extract(input_data=url, query=query)

# Check the results
from axetract.data_types import Status

if result.status == Status.SUCCESS:
    print("Extracted Data:", result.prediction)
    print("Source XPaths:", result.xpaths)
else:
    print("Extraction Failed:", result.error)
```

## How it works

1. **Preprocessing**: The pipeline fetches the HTML from the URL and chunks it.
2. **Pruning**: The Pruner adapter filters the HTML chunks to keep only those relevant to "price, model name, and color".
3. **Extraction**: The Extractor adapter analyzes the pruned HTML and generates the structured JSON output.
4. **GXR**: The results include XPaths that point back to the exact location of the values in the original DOM.
