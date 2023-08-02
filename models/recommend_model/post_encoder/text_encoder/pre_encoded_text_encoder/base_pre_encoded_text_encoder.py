from models.recommend_model.post_encoder.text_encoder.base_text_encoder import BaseTextEncoder
from abc import ABC, abstractmethod
from torch import Tensor


class BasePreEncodedTextEncoder(BaseTextEncoder, ABC):
    def __init__(self, out_features: int, in_features: int):
        super(BasePreEncodedTextEncoder, self).__init__(out_features=out_features)
        self.in_features: int = in_features
        
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        '''
        x: batch_size, in_features
        return: batch_size, out_features
        '''
        pass
