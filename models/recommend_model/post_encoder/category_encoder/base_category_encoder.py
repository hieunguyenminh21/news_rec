from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor


class BaseCategoryEncoder(nn.Module, ABC):
    def __init__(self, out_features: int):
        super(BaseCategoryEncoder, self).__init__()
        self.out_features: int = out_features

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        '''

        :param x: batch_size
        :return: batch_size, out_features
        '''
        pass

