from axetract import AXECleaner, AXEPruner,
from axetract.cleaners import Trafilatura
from litellm import LiteLLM


pipeline = AXEpipeline(
 LiteLLM(some parameters)
)

pipeline = AXEpipeline(
 cleaner=Trafilatura()
)

pipeline = AXEpipeline()