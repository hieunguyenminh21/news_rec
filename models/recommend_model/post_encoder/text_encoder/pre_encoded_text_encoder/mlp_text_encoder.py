from .base_pre_encoded_text_encoder import BasePreEncodedTextEncoder
import torch.nn as nn
from torch import Tensor


class MLPTextEncoder(BasePreEncodedTextEncoder):
    def __init__(self, out_features: int, in_features: int):
        super(MLPTextEncoder, self).__init__(out_features=out_features, in_features=in_features)
        self.core = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features),
                                  nn.ReLU(inplace=True))
        
    def forward(self, x: Tensor) -> Tensor:
        '''
        x: batch_size, in_features
        return: batch_size, out_features
        '''
        return self.core(x)
