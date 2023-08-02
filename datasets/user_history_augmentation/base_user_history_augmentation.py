from abc import ABC, abstractmethod
from typing import List


class BaseUserHistoryAugmentation(ABC):
    @abstractmethod
    def augment(self, history: List[int]) -> List[int]:
        pass
