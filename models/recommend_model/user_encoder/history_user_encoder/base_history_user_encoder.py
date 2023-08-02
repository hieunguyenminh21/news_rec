from models.recommend_model.user_encoder.base_user_encoder import BaseUserEncoder
from abc import ABC, abstractmethod
from torch import Tensor


class BaseHistoryUserEncoder(BaseUserEncoder, ABC):
    @abstractmethod
    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''
        :param x: batch_size, sequence_len, out_features
        :param attention_mask: batch_size, sequence_len
        :return: batch_size, out_features
        '''
        pass
