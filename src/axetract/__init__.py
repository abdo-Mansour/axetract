"""AXEtract: Low-Cost Cross-Domain Web Structured Information Extraction."""

from importlib.metadata import PackageNotFoundError, version

from axetract.data_types import AXEResult, AXESample, Status
from axetract.pipeline import AXEPipeline

try:
    __version__ = version("axetract")
except PackageNotFoundError:  # editable install not yet built
    __version__ = "unknown"

__all__ = ["AXEPipeline", "AXESample", "AXEResult", "Status", "__version__"]
