from torch import Tensor
from .base_category_encoder import BaseCategoryEncoder
import torch.nn as nn


class MLPCategoryEncoder(BaseCategoryEncoder):
    def __init__(self, out_features: int, category_embedding: nn.Embedding):
        super(MLPCategoryEncoder, self).__init__(out_features=out_features)
        self.category_embedding: nn.Embedding = category_embedding
        self.process = nn.Sequential(nn.Linear(in_features=category_embedding.embedding_dim, out_features=out_features),
                                     nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        '''

        :param x: batch_size
        :return: batch_size, out_features
        '''
        embedding = self.category_embedding(x)      # batch_size, embedding_dim
        encode = self.process(embedding)    # batch_size, out_features

        return encode
