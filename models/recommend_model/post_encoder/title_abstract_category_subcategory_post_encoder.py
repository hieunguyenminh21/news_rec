from .base_post_encoder import BasePostEncoder
import torch
from .text_encoder import BaseTextEncoder, BaseRawTextEncoder, BasePreEncodedTextEncoder
from .category_encoder import BaseCategoryEncoder
from typing import Dict
from torch import Tensor
from models.layers import AdditiveAttention


class TitleAbstractCategorySubcategoryPostEncoder(BasePostEncoder):
    def __init__(self, out_features: int, title_encoder: BaseTextEncoder, abstract_encoder: BaseTextEncoder,
                 category_encoder: BaseCategoryEncoder, subcategory_encoder: BaseCategoryEncoder, query_vector_dim: int):
        super(TitleAbstractCategorySubcategoryPostEncoder, self).__init__(out_features=out_features)
        self.title_encoder: BaseTextEncoder = title_encoder
        self.abstract_encoder: BaseTextEncoder = abstract_encoder
        self.category_encoder: BaseCategoryEncoder = category_encoder
        self.subcategory_encoder: BaseCategoryEncoder = subcategory_encoder
        self.view_attention = AdditiveAttention(in_features=out_features, query_vector_dim=query_vector_dim)

    def get_title_encode(self, data: Dict):
        if isinstance(self.title_encoder, BaseRawTextEncoder):
            title_token_ids: Tensor = data["title_token_ids"]
            title_attention_mask: Tensor = data["title_attention_mask"]
            return self.title_encoder(token_ids=title_token_ids, attention_mask=title_attention_mask)
        
        elif isinstance(self.title_encoder, BasePreEncodedTextEncoder):
            title_pre_encode: Tensor = data["title_pre_encode"]
            return self.title_encoder(x=title_pre_encode)
        
        else:
            raise ValueError("Invalid title_encoder class")

    def get_abstract_encode(self, data: Dict):
        if isinstance(self.abstract_encoder, BaseRawTextEncoder):
            abstract_token_ids: Tensor = data["abstract_token_ids"]
            abstract_attention_mask: Tensor = data["abstract_attention_mask"]
            return self.abstract_encoder(token_ids=abstract_token_ids, attention_mask=abstract_attention_mask)
        
        elif isinstance(self.abstract_encoder, BasePreEncodedTextEncoder):
            abstract_pre_encode: Tensor = data["abstract_pre_encode"]
            return self.abstract_encoder(x=abstract_pre_encode)
        
        else:
            raise ValueError("Invalid abstract_encoder class")

    def get_category_encode(self, data):
        category: Tensor = data["category"]
        return self.category_encoder(x=category)

    def get_subcategory_encode(self, data):
        subcategory: Tensor = data["subcategory"]
        return self.subcategory_encoder(x=subcategory)

    def forward(self, data: Dict) -> Tensor:
        title_encode = self.get_title_encode(data=data)
        abstract_encode = self.get_abstract_encode(data=data)
        category_encode = self.get_category_encode(data=data)
        subcategory_encode = self.get_subcategory_encode(data=data)
        
        encode_matrix = torch.stack([title_encode, abstract_encode, category_encode, subcategory_encode], dim=1)
        context = self.view_attention(x=encode_matrix)

        return context