from typing import List
from .base_user_history_augmentation import BaseUserHistoryAugmentation
from .user_history_item_crop import UserHistoryItemCrop
from .user_history_item_mask import UserHistoryItemMask
from .user_history_item_reorder import UserHistoryItemReorder
from .user_history_item_insert import UserHistoryItemInsert
from .user_history_item_substitute import UserHistoryItemSubstitute
import random
from typing import Dict


class UserHistoryAugmentationVersion2(BaseUserHistoryAugmentation):
    def __init__(self, crop_rate: float, mask_rate: float, reorder_rate: float, insert_rate: float, substitute_rate: float,
                 item_to_similar_item: Dict[int, int], history_length_threshold: int):
        super(UserHistoryAugmentationVersion2, self).__init__()
        self.crop_augmenter: BaseUserHistoryAugmentation = UserHistoryItemCrop(crop_rate=crop_rate)
        self.mask_augmenter: BaseUserHistoryAugmentation = UserHistoryItemMask(mask_rate=mask_rate)
        self.reorder_augmenter: BaseUserHistoryAugmentation = UserHistoryItemReorder(reorder_rate=reorder_rate)
        self.insert_augmenter: BaseUserHistoryAugmentation = UserHistoryItemInsert(insert_rate=insert_rate,
                                                                                   item_to_similar_item=item_to_similar_item)
        self.substitute_augmenter: BaseUserHistoryAugmentation = UserHistoryItemSubstitute(substitute_rate=substitute_rate,
                                                                                           item_to_similar_item=item_to_similar_item)
        self.history_length_threshold: int = history_length_threshold

    def augment(self, history: List[int]) -> List[int]:
        if len(history) <= self.history_length_threshold:
            augmenter: BaseUserHistoryAugmentation = random.choice([self.insert_augmenter, self.substitute_augmenter])
        else:
            augmenter: BaseUserHistoryAugmentation = random.choice([self.crop_augmenter, self.mask_augmenter, self.reorder_augmenter,
                                                                    self.insert_augmenter, self.substitute_augmenter])
        return augmenter.augment(history=history)
