from abc import ABC
import torch.nn as nn


class BaseUserEncoder(nn.Module, ABC):
    def __init__(self, out_features: int):
        super(BaseUserEncoder, self).__init__()
        self.out_features: int = out_features
