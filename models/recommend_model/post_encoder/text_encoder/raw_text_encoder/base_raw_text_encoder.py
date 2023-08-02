from models.recommend_model.post_encoder.text_encoder.base_text_encoder import BaseTextEncoder
from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn as nn


class BaseRawTextEncoder(BaseTextEncoder, ABC):
    def __init__(self, out_features: int, word_embedding: nn.Embedding):
        super(BaseRawTextEncoder, self).__init__(out_features=out_features)
        self.word_embedding: nn.Embedding = word_embedding
        
    @abstractmethod
    def forward(self, token_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''
        :param token_ids: batch_size, length
        :param attention_mask: batch_size, length
        :return: batch_size, out_features
        '''
        pass
