from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from torch import Tensor
import numpy as np
from scipy.sparse import csr_matrix
import torch
from .user_history_augmentation import BaseUserHistoryAugmentation
import random


class RecommendDataset(Dataset):
    def __init__(self, post_id_to_info: Dict[int, Dict], behaviours: List[Dict], num_categories: int, history_length: int, 
                 num_negative_samples: int, use_pre_encoded_text: bool, use_side_feature: bool,
                 use_contrastive_learning: bool, user_history_augmenter: BaseUserHistoryAugmentation):
        super(RecommendDataset, self).__init__()
        self.post_id_to_info: Dict[int, Dict] = post_id_to_info
        
        self.list_history: List[List[int]] = list(map(lambda x: x['history'], behaviours))
        self.list_negative: List[List[int]] = list(map(lambda x: x['negative'], behaviours))
        self.list_positive_index: List[Tuple[int, int]] = []
        for index, behaviour in enumerate(behaviours):
            for post_id in behaviour["positive"]:
                self.list_positive_index.append((index, post_id))
        
        self.num_categories: int = num_categories
        self.history_length: int = history_length
        self.num_negative_samples: int = num_negative_samples
        self.use_pre_encoded_text: bool = use_pre_encoded_text
        self.use_side_feature: bool = use_side_feature
        self.use_contrastive_learning: bool = use_contrastive_learning
        self.user_history_augmenter: BaseUserHistoryAugmentation = user_history_augmenter
        
    def __len__(self) -> int:
        return len(self.list_positive_index)
    
    def get_list_posts_features(self, post_ids: List[int]) -> Dict[str, Tensor]:
        def _convert_numpy_to_tensor(_data: np.ndarray) -> Tensor:
            return torch.tensor(_data, dtype=torch.float)
        
        def _convert_csr_matrix_to_tensor(_data: csr_matrix) -> Tensor:
            return torch.tensor(_data.toarray(), dtype=torch.float).squeeze(dim=0)
        
        result: Dict[str, Tensor] = {}
        
        posts_info: List[Dict] = list(map(lambda x: self.post_id_to_info[x], post_ids))
        
        if self.use_pre_encoded_text:
            titles: List[np.ndarray] = list(map(lambda x: x["title_bert_encode"], posts_info))
            titles: List[Tensor] = list(map(_convert_numpy_to_tensor, titles))
            title: Tensor = torch.stack(titles, dim=0)
            result["title_pre_encode"] = title

            abstracts: List[np.ndarray] = list(map(lambda x: x["abstract_bert_encode"], posts_info))
            abstracts: List[Tensor] = list(map(_convert_numpy_to_tensor, abstracts))
            abstract: Tensor = torch.stack(abstracts, dim=0)
            result["abstract_pre_encode"] = abstract
        
        else:
            title_token_ids: List[List[int]] = list(map(lambda x: x["title_token_ids"], posts_info))
            title_token_ids: Tensor = torch.tensor(title_token_ids, dtype=torch.int)
            title_attention_mask: Tensor = (title_token_ids != 0).float()
            result["title_token_ids"] = title_token_ids
            result["title_attention_mask"] = title_attention_mask
            
            abstract_token_ids: List[List[int]] = list(map(lambda x: x["abstract_token_ids"], posts_info))
            abstract_token_ids: Tensor = torch.tensor(abstract_token_ids, dtype=torch.int)
            abstract_attention_mask: Tensor = (abstract_token_ids != 0).float()
            result["abstract_token_ids"] = abstract_token_ids
            result["abstract_attention_mask"] = abstract_attention_mask
        
        categories: List[int] = list(map(lambda x: x["category_id"], posts_info))
        category: Tensor = torch.tensor(categories, dtype=torch.int)
        result["category"] = category
        
        subcategories: List[int] = list(map(lambda x: x["subcategory_id"], posts_info))
        subcategory: Tensor = torch.tensor(subcategories, dtype=torch.int)
        result["subcategory"] = subcategory
        
        return result
    
    def get_padded_history(self, history: List[int]) -> List[int]:
        history: List[int] = history[:self.history_length]
        history: List[int] = history + list(map(lambda x: 0, range(self.history_length - len(history))))
        assert len(history) == self.history_length
        return history
    
    def get_user_side_feature(self, history: List[int]) -> Tensor:
        side_feature: np.ndarray = np.ones(self.num_categories-1)
        count: int = 0
        for post_id in history:
            info: Dict = self.post_id_to_info[post_id]
            category_id: int = info["category_id"]
            subcategory_id: int = info["subcategory_id"]
            if category_id != 0 or subcategory_id != 0:
                count += 1
                if category_id != 0:
                    side_feature[category_id-1] = side_feature[category_id-1] + 1.0
                if subcategory_id != 0:
                    side_feature[subcategory_id-1] = side_feature[subcategory_id-1] + 1.0
        if count > 0:
            side_feature = side_feature / count
        return torch.tensor(side_feature, dtype=torch.float)
    
    def sample_negative(self, negative: List[int]) -> List[int]:
        if len(negative) >= self.num_negative_samples:
            return list(random.sample(negative, k=self.num_negative_samples))
        else:
            return list(random.choices(negative, k=self.num_negative_samples))
    
    def __getitem__(self, index: int) -> Dict:
        result: Dict = {}
        
        index, post_id = self.list_positive_index[index]
        positive: List[int] = [post_id]
        negative: List[int] = self.sample_negative(negative=self.list_negative[index])
        history: List[int] = self.list_history[index]        
        
        candidate: List[int] = positive + negative
        candidate_label: List[int] = list(map(lambda x: 1, positive)) + list(map(lambda x: 0, negative))
        
        candidate_features: Dict[str, Tensor] = self.get_list_posts_features(post_ids=candidate)
        candidate_label: Tensor = torch.tensor(candidate_label, dtype=torch.float)
        
        result["candidate_label"] = candidate_label
        result["candidate_features"] = candidate_features
        
        if self.use_side_feature:
            side_feature: Tensor = self.get_user_side_feature(history=history)
            result["side_feature"] = side_feature
            
        padded_history: List[int] = self.get_padded_history(history=history)
        
        history_features: Dict[str, Tensor] = self.get_list_posts_features(post_ids=padded_history)
        history_attn_mask: Tensor = torch.tensor(list(map(lambda x: 0 if x==0 else 1, padded_history)), dtype=torch.float)
        
        result["history_attn_mask"] = history_attn_mask
        result["history_features"] = history_features
        
        if self.use_contrastive_learning:
            history_1: List[int] = self.user_history_augmenter.augment(history=history)
            if self.use_side_feature:
                side_feature_1: Tensor = self.get_user_side_feature(history=history_1)
                result["side_feature_1"] = side_feature_1
                
            padded_history_1: List[int] = self.get_padded_history(history=history_1)
            
            history_features_1: Dict[str, Tensor] = self.get_list_posts_features(post_ids=padded_history_1)
            history_attn_mask_1: Tensor = torch.tensor(list(map(lambda x: 0 if x==0 else 1, padded_history_1)), dtype=torch.float)
            
            result["history_attn_mask_1"] = history_attn_mask_1
            result["history_features_1"] = history_features_1
            
            history_2: List[int] = self.user_history_augmenter.augment(history=history)
            
            if self.use_side_feature:
                side_feature_2: Tensor = self.get_user_side_feature(history=history_2)
                result["side_feature_2"] = side_feature_2
                
            padded_history_2: List[int] = self.get_padded_history(history=history_2)
            
            history_features_2: Dict[str, Tensor] = self.get_list_posts_features(post_ids=padded_history_2)
            history_attn_mask_2: Tensor = torch.tensor(list(map(lambda x: 0 if x==0 else 1, padded_history_2)), dtype=torch.float)
            
            result["history_attn_mask_2"] = history_attn_mask_2
            result["history_features_2"] = history_features_2
            
        return result