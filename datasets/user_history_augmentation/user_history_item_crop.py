from typing import List
from .base_user_history_augmentation import BaseUserHistoryAugmentation
import random


class UserHistoryItemCrop(BaseUserHistoryAugmentation):
    def __init__(self, crop_rate: float):
        super(UserHistoryItemCrop, self).__init__()
        self.crop_rate: float = crop_rate

    def augment(self, history: List[int]) -> List[int]:
        if not history:
            return history
        crop_length: int = max(int(self.crop_rate * len(history)), 1)
        start_index: int = random.randint(0, len(history) - crop_length)
        return history[start_index:start_index+crop_length]
