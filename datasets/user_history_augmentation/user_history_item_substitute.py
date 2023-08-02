from typing import List, Dict, Set
import random
from .base_user_history_augmentation import BaseUserHistoryAugmentation


class UserHistoryItemSubstitute(BaseUserHistoryAugmentation):
    def __init__(self, substitute_rate: float, item_to_similar_item: Dict[int, int]):
        super(UserHistoryItemSubstitute, self).__init__()
        self.substitute_rate: float = substitute_rate
        self.item_to_similar_item: Dict[int, int] = item_to_similar_item

    def augment(self, history: List[int]) -> List[int]:
        copied_history: List[int] = history.copy()
        substitute_length: int = int(self.substitute_rate * len(copied_history))
        substitute_indexes: Set[int] = set(random.sample(range(len(copied_history)), k=substitute_length))
        for index in substitute_indexes:
            copied_history[index] = self.item_to_similar_item[history[index]]
        return copied_history
