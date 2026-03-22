from abc import ABC, abstractmethod
from typing import List

from axetract.data_types import AXESample


class BasePruner(ABC):
    """Abstract base class for all pruners."""

    def __init__(self, name: str):
        """Initialize the pruner.

        Args:
            name (str): Component name.
        """
        self.name = name

    @abstractmethod
    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        """Prune a batch of samples.

        Args:
            samples (List[AXESample]): Input samples.

        Returns:
            List[AXESample]: Pruned samples.
        """
        raise NotImplementedError
