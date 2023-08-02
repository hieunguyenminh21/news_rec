import torch.nn as nn
from .post_encoder import BasePostEncoder
from .user_encoder import BaseUserEncoder, BaseHistoryUserEncoder
from typing import Dict
from torch import Tensor


class RecommendModel(nn.Module):
    def __init__(self, out_features: int, post_encoder: BasePostEncoder, user_encoder: BaseUserEncoder):
        super(RecommendModel, self).__init__()
        
        self.post_encoder: PostEncoder = post_encoder
        self.user_encoder: UserEncoder = user_encoder
    
    def compute_batch_posts_encodes(self, data: Dict[str, Tensor]) -> Tensor:
        return self.post_encoder(data=data)
    
    def compute_batch_list_posts_encodes(self, data: Dict[str, Tensor]) -> Tensor:
        batch_size: int = None
        length: int = None
        
        flatten_data: Dict[str, Tensor] = {}
        for key, value in data.items():
            if batch_size is None:
                batch_size, length = value.shape[:2]
            else:
                assert batch_size == value.shape[0] and length == value.shape[1]
                
            flatten_data[key] = value.reshape((batch_size*length,) + value.shape[2:])  
        
        flatten_encode: Tensor = self.compute_batch_posts_encodes(data=flatten_data)     # batch_size * length, out_features
        return flatten_encode.reshape(batch_size, length, -1)
    
    def compute_batch_users_encodes(self, history: Tensor, history_attn_mask: Tensor, side_feature: Tensor = None) -> Tensor:
        if isinstance(self.user_encoder, BaseHistoryUserEncoder):
            return self.user_encoder(x=history, attention_mask=history_attn_mask)
        else:
            raise ValueError("Invalid data type for user encoder")
    
    def compute_batch_scores(self, user_encode: Tensor, candidate_encode: Tensor) -> Tensor:
        user_encode = user_encode.unsqueeze(dim=1)    # batch_size, 1, out_features
        
        score = user_encode * candidate_encode     # batch_size, num_candidates, out_features
        score = score.sum(dim=2, keepdim=False)    # batch_size, num_candidates
        
        return score
    
    def forward(self, candidate: Dict[str, Tensor], history: Dict[str, Tensor], history_attn_mask: Tensor, side_feature: Tensor = None) -> Tensor:
        candidate_encode = self.compute_batch_list_posts_encodes(candidate)
        history_encode = self.compute_batch_list_posts_encodes(history)
        user_encode = self.compute_batch_users_encodes(history=history_encode, history_attn_mask=history_attn_mask, side_feature=side_feature)
        score = self.compute_batch_scores(user_encode=user_encode, candidate_encode=candidate_encode)
        return score
        