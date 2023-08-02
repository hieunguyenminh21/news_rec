import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict


class BasePostEncoder(ABC, nn.Module):
    def __init__(self, out_features: int):
        super(BasePostEncoder, self).__init__()
        self.out_features: int = out_features
        
    def forward(self, data: Dict):
        pass
