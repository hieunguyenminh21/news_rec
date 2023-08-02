import random
from typing import List, Set, Dict
from .base_user_history_augmentation import BaseUserHistoryAugmentation


class UserHistoryItemInsert(BaseUserHistoryAugmentation):
    def __init__(self, insert_rate: float, item_to_similar_item: Dict[int, int]):
        super(UserHistoryItemInsert, self).__init__()
        self.insert_rate: float = insert_rate
        self.item_to_similar_item: Dict[int, int] = item_to_similar_item

    def augment(self, history: List[int]) -> List[int]:
        insert_length: int = int(self.insert_rate*len(history))
        insert_indexes: Set[int] = set(random.sample(range(len(history)), k=insert_length))
        result: List[int] = []
        for index, item in enumerate(history):
            if index in insert_indexes:
                result.append(self.item_to_similar_item[item])
            result.append(item)
        return result
