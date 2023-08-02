from typing import List
from .base_user_history_augmentation import BaseUserHistoryAugmentation
from .user_history_item_crop import UserHistoryItemCrop
from .user_history_item_mask import UserHistoryItemMask
from .user_history_item_reorder import UserHistoryItemReorder
import random


class UserHistoryAugmentationVersion1(BaseUserHistoryAugmentation):
    def __init__(self, crop_rate: float, mask_rate: float, reorder_rate: float):
        super(UserHistoryAugmentationVersion1, self).__init__()
        self.crop_augmenter: BaseUserHistoryAugmentation = UserHistoryItemCrop(crop_rate=crop_rate)
        self.mask_augmenter: BaseUserHistoryAugmentation = UserHistoryItemMask(mask_rate=mask_rate)
        self.reorder_augmenter: BaseUserHistoryAugmentation = UserHistoryItemReorder(reorder_rate=reorder_rate)

    def augment(self, history: List[int]) -> List[int]:
        augmenter: BaseUserHistoryAugmentation = random.choice([self.crop_augmenter, self.mask_augmenter, self.reorder_augmenter])
        return augmenter.augment(history=history)
