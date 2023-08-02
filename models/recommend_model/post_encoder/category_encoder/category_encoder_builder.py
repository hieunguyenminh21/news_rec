from .base_category_encoder import BaseCategoryEncoder
from .mlp_category_encoder import MLPCategoryEncoder
from typing import Dict
import torch.nn as nn


class CategoryEncoderBuilder:
    @classmethod
    def build_category_encoder(cls, config: Dict, **kwargs) -> BaseCategoryEncoder:
        out_features: int = config["hidden_dim"]
        if config["category_encoder_method"] == "MLP":
            category_embedding: nn.Embedding = kwargs["category_embedding"]
            return MLPCategoryEncoder(out_features=out_features, category_embedding=category_embedding)
        else:
            raise ValueError(f"Invalid category_encoder_method, got: {config['category_encoder_method']}")