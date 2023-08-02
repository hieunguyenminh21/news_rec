from .base_raw_text_encoder import BaseRawTextEncoder
from .cnn_text_encoder import CNNTextEncoder
from .fastformer_text_encoder import FastformerTextEncoder
from .multi_head_self_attention_text_encoder import MultiHeadSelfAttentionTextEncoder
from .transformer_text_encoder import TransformerTextEncoder
from typing import Dict
import torch.nn as nn


class RawTextEncoderBuilder:
    @classmethod
    def build_raw_text_encoder(cls, config: Dict, word_embedding: nn.Embedding) -> BaseRawTextEncoder:
        out_features: int = config["hidden_dim"]
        
        if config["raw_text_encoder_method"] == "CNN":
            kernel_size: int = config["raw_text_encoder_cnn_kernel_size"]
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            drop_out: float = config["drop_out"]
            return CNNTextEncoder(out_features=out_features, word_embedding=word_embedding, 
                                  kernel_size=kernel_size, query_vector_dim=query_vector_dim, drop_out=drop_out)
        
        elif config["raw_text_encoder_method"] == "Fastformer":
            num_heads: int = config["raw_text_encoder_fastformer_num_heads"]
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            drop_out: float = config["drop_out"]
            return FastformerTextEncoder(out_features=out_features, word_embedding=word_embedding, 
                                         num_heads=num_heads, query_vector_dim=query_vector_dim, drop_out=drop_out)
        
        elif config["raw_text_encoder_method"] == "MultiHeadSelfAttention":
            num_heads: int = config["raw_text_encoder_multi_head_self_attention_num_heads"]
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            drop_out: float = config["drop_out"]
            return MultiHeadSelfAttentionTextEncoder(out_features=out_features, word_embedding=word_embedding, 
                                                     num_heads=num_heads, query_vector_dim=query_vector_dim, drop_out=drop_out)
        
        elif config["raw_text_encoder_method"] == "Transformer":
            num_heads: int = config["raw_text_encoder_transformer_num_heads"]
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            drop_out: float = config["drop_out"]
            return TransformerTextEncoder(out_features=out_features, word_embedding=word_embedding, 
                                          num_heads=num_heads, query_vector_dim=query_vector_dim, drop_out=drop_out)

        else:
            raise ValueError(f"Invalid raw_text_encoder_method, got: {config['raw_text_encoder_method']}")