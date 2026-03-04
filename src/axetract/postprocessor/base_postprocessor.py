from abc import ABC, abstractmethod
from typing import List

from axetract.data_types import AXESample


class BasePostprocessor(ABC):
    """Abstract base class for all postprocessors."""

    def __init__(self, name: str):
        """Initialize the postprocessor.

        Args:
            name (str): Component name.
        """
        self.name = name

    @abstractmethod
    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        """Postprocess a batch of samples.

        Args:
            samples (List[AXESample]): Input samples.

        Returns:
            List[AXESample]: Postprocessed samples.
        """
        raise NotImplementedError
