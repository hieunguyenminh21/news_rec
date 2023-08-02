from abc import ABC
import torch.nn as nn


class BaseTextEncoder(ABC, nn.Module):
    def __init__(self, out_features: int):
        super(BaseTextEncoder, self).__init__()
        self.out_features: int = out_features
