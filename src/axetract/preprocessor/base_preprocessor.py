from abc import ABC, abstractmethod
from typing import List

from axetract.data_types import AXESample


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""

    def __init__(self, name: str):
        """Initialize the preprocessor.

        Args:
            name (str): Component name.
        """
        self.name = name

    @abstractmethod
    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        """Process a batch of samples.

        Args:
            samples (List[AXESample]): Input samples.

        Returns:
            List[AXESample]: Processed samples.
        """
        raise NotImplementedError
