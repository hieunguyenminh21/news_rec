import random
from typing import List
from .base_user_history_augmentation import BaseUserHistoryAugmentation


class UserHistoryItemMask(BaseUserHistoryAugmentation):
    def __init__(self, mask_rate: float, mask_value: int = 0):
        super(UserHistoryItemMask, self).__init__()
        self.mask_rate: float = mask_rate
        self.mask_value: int = mask_value

    def augment(self, history: List[int]) -> List[int]:
        copied_history: List[int] = history.copy()
        mask_length: int = int(self.mask_rate*len(copied_history))
        mask_indexes: List[int] = random.sample(range(len(copied_history)), k=mask_length)
        for index in mask_indexes:
            copied_history[index] = self.mask_value
        return copied_history

