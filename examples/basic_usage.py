# Examples on how to use Axetract package

# > local means it's ran with local models running with transformers libraries with our adaptors and fine tuned qwen model

# %%
## Basic AXE usage with natural language query (local model ran via transformers and HF models as well as AXE Adaptors)
from axetract import AXEPipeline

extractor = AXEPipeline.from_config()
result = extractor.extract(
    input_data="https://en.wikipedia.org/wiki/Alija_Izetbegovi%C4%87",
    query="When was he born?"
)
print(result.prediction)



# %%
## Basic usage with schema query (local)
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
