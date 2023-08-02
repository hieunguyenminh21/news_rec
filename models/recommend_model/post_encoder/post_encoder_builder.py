from .base_post_encoder import BasePostEncoder
from .title_abstract_category_subcategory_post_encoder import TitleAbstractCategorySubcategoryPostEncoder
from .text_encoder import BaseTextEncoder, TextEncoderBuilder
from .category_encoder import BaseCategoryEncoder, CategoryEncoderBuilder
from typing import Dict, Tuple
import torch.nn as nn
from common.utils import PickleReadObjectFromLocalPatient
import numpy as np
import torch
from torch import Tensor


class PostEncoderBuilder:
    @classmethod
    def load_pretrained_word_embedding(cls, config: Dict) -> nn.Embedding:
        data_dir: str = config["data_dir"]
        object_reader = PickleReadObjectFromLocalPatient()
        embedding_matrix: np.ndarray = object_reader.read(file_name=f"{data_dir}/pretrained/title_abstract_word_embedding.pkl", num_tries=1, wait_time=0.0)
        embedding_matrix: Tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix, padding_idx=0, freeze=False)
        return word_embedding
    
    @classmethod
    def build_category_embedding(cls, config: Dict) -> nn.Embedding:
        category_embedding_dim: int = config["category_embedding_dim"]
        num_categories: int = config["num_categories"]
        category_embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=category_embedding_dim, padding_idx=0)
        return category_embedding

    @classmethod
    def build_post_encoder(cls, config: Dict) -> BasePostEncoder:
        out_features: int = config["hidden_dim"]
        
        if config["post_encoder_method"] == "TitleAbtractCategorySubcategory":
            if config["text_encoder_method"] == "Raw":
                word_embedding: nn.Embedding = cls.load_pretrained_word_embedding(config=config)
                title_encoder: BaseTextEncoder = TextEncoderBuilder.build_text_encoder(config=config, word_embedding=word_embedding)
                abstract_encoder: BaseTextEncoder = TextEncoderBuilder.build_text_encoder(config=config, word_embedding=word_embedding)
            else:
                title_encoder: BaseTextEncoder = TextEncoderBuilder.build_text_encoder(config=config)
                abstract_encoder: BaseTextEncoder = TextEncoderBuilder.build_text_encoder(config=config)
            
            category_embedding: nn.Embedding = cls.build_category_embedding(config=config)
            category_encoder: BaseCategoryEncoder = CategoryEncoderBuilder.build_category_encoder(config=config, category_embedding=category_embedding)
            subcategory_encoder: BaseCategoryEncoder = CategoryEncoderBuilder.build_category_encoder(config=config, category_embedding=category_embedding)
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            return TitleAbstractCategorySubcategoryPostEncoder(out_features=out_features, title_encoder=title_encoder, 
                                                               abstract_encoder=abstract_encoder,
                                                               category_encoder=category_encoder, subcategory_encoder=subcategory_encoder,
                                                               query_vector_dim=query_vector_dim)
        else:
            raise ValueError(f"Invalid post_encoder_method, got: {config['post_encoder_method']}")