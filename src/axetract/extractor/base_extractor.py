from abc import ABC, abstractmethod
from typing import List

from axetract.data_types import AXESample


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self, name: str):
        """Initialize the extractor.

        Args:
            name (str): Component name.
        """
        self.name = name

    @abstractmethod
    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        """Extract data from a batch of samples.

        Args:
            samples (List[AXESample]): Input samples.

        Returns:
            List[AXESample]: Samples with predictions.
        """
        raise NotImplementedError
