from abc import ABC, abstractmethod
from axetract.data_types import AXESample
from typing import List

class BasePreprocessor(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, samples: List[AXESample]) -> List[AXESample]:
        raise NotImplementedError