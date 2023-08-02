from .base_user_history_augmentation import BaseUserHistoryAugmentation
from .user_history_item_crop import UserHistoryItemCrop
from .user_history_item_insert import UserHistoryItemInsert
from .user_history_item_mask import UserHistoryItemMask
from .user_history_item_reorder import UserHistoryItemReorder
from .user_history_item_substitute import UserHistoryItemSubstitute
from .user_history_augmentation_version_1 import UserHistoryAugmentationVersion1
from .user_history_augmentation_version_2 import UserHistoryAugmentationVersion2
from typing import Dict
from common.utils import PickleReadObjectFromLocalPatient


class UserHistoryAugmentationBuilder:
    @classmethod
    def build_user_history_augmentation(cls, config: Dict, data_split_dir: str = None) -> BaseUserHistoryAugmentation:
        if config["user_history_augmentation_method"] == "Crop":
            return UserHistoryItemCrop(crop_rate=config["crop_rate"])
        elif config["user_history_augmentation_method"] == "Insert":
            item_to_similar_item: Dict[int, int] = PickleReadObjectFromLocalPatient().read(file_name=f"{data_split_dir}/item_to_similar_item.pkl")
            return UserHistoryItemInsert(insert_rate=config["insert_rate"], item_to_similar_item=item_to_similar_item)
        elif config["user_history_augmentation_method"] == "Mask":
            return UserHistoryItemMask(mask_rate=config["mask_rate"])
        elif config["user_history_augmentation_method"] == "Reorder":
            return UserHistoryItemReorder(reorder_rate=config["reorder_rate"])
        elif config["user_history_augmentation_method"] == "Substitute":
            item_to_similar_item: Dict[int, int] = PickleReadObjectFromLocalPatient().read(file_name=f"{data_split_dir}/item_to_similar_item.pkl")
            return UserHistoryItemSubstitute(substitute_rate=config["substitute_rate"], item_to_similar_item=item_to_similar_item)
        elif config["user_history_augmentation_method"] == "Version1":
            return UserHistoryAugmentationVersion1(crop_rate=config["crop_rate"], mask_rate=config["mask_rate"], reorder_rate=config["reorder_rate"])
        
        elif config["user_history_augmentation_method"] == "Version2":
            item_to_similar_item: Dict[int, int] = PickleReadObjectFromLocalPatient().read(file_name=f"{data_split_dir}/item_to_similar_item.pkl")
            return UserHistoryAugmentationVersion2(crop_rate=config["crop_rate"], mask_rate=config["mask_rate"], reorder_rate=config["reorder_rate"],
                                                   insert_rate=config["insert_rate"], substitute_rate=config["substitute_rate"],
                                                   item_to_similar_item=item_to_similar_item, history_length_threshold=config["history_length_threshold"])
        else:
            raise ValueError("Invalid user_history_augmentation_method")
