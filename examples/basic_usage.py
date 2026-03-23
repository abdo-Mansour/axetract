# ruff: noqa D101

# Examples on how to use Axetract package

# > local means it's ran with local models running with transformers libraries with our adaptors and fine tuned qwen model

# %% Basic AXE usage with natural language query
from axetract import AXEPipeline

extractor = AXEPipeline.from_config()
result = extractor.extract(
    input_data="https://en.wikipedia.org/wiki/Alija_Izetbegovi%C4%87", # or you could use Path("path/to/local/page.html")
    query="When was he born?"
)
print(result.prediction)

# %% Basic usage with schema query (local)
from pydantic import BaseModel

from axetract import AXEPipeline


class PersonSchema(BaseModel):
    """Schema for extracting person information."""
    name: str
    birth_date: str
    death_date: str = ""
    occupation: str
    nationality: str


extractor = AXEPipeline.from_config()
result = extractor.extract(
    input_data="https://en.wikipedia.org/wiki/Alija_Izetbegovi%C4%87",
    schema=PersonSchema,
)
print(result.prediction)



# %%
# Batch Processing with Same Query (Multiple URLs)

from axetract import AXEPipeline

extractor = AXEPipeline.from_config()
urls = [
    "https://en.wikipedia.org/wiki/Alija_Izetbegovi%C4%87",
    "https://en.wikipedia.org/wiki/Izz_ad-Din_al-Qassam",
    "https://en.wikipedia.org/wiki/Nelson_Mandela",
]
results = extractor.extract(
    urls,
    query="Extract the person's name, birth date, and occupation"
)
for r in results:
    print(r.prediction)

# %%Use vLLM for high-throughput serving
from axetract import AXEPipeline

extractor = AXEPipeline.from_config(
    use_vllm=True,
    vllm_base_url="http://localhost:8000"
)
result = extractor.extract(
    input_data="https://example.com/product",
    query="Extract product name, price, and availability"
)

# %% Batch of different 
from pydantic import BaseModel

from axetract import AXEPipeline
from axetract.data_types import AXESample


class PersonSchema(BaseModel):
    name: str
    birth_date: str

class ProductSchema(BaseModel): 
    name: str
    price: float

pipeline = AXEPipeline.from_config()

batch = [
    AXESample(
        content="https://en.wikipedia.org/wiki/Albert_Einstein",
        schema_model=PersonSchema,
    ),
    AXESample(
        content="https://example.com/product/123",
        schema_model=ProductSchema,
    ),
    AXESample(
        content="https://example.com/article",
        query="Extract the article title and author",
    ),
]

results = pipeline.extract_batch(batch)
