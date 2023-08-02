import random
from typing import List
from .base_user_history_augmentation import BaseUserHistoryAugmentation


class UserHistoryItemReorder(BaseUserHistoryAugmentation):
    def __init__(self, reorder_rate: float):
        super(UserHistoryItemReorder, self).__init__()
        self.reorder_rate: float = reorder_rate

    def augment(self, history: List[int]) -> List[int]:
        if len(history) < 2:
            return history
        reorder_length: int = max(int(self.reorder_rate*len(history)), 2)
        start_index: int = random.randint(0, len(history) - reorder_length)
        prefix: List[int] = history[:start_index]
        middle: List[int] = history[start_index:start_index+reorder_length]
        suffix: List[int] = history[start_index+reorder_length:]
        random.shuffle(middle)
        return prefix + middle + suffix
