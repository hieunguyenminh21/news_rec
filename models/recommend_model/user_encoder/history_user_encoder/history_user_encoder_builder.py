from .base_history_user_encoder import BaseHistoryUserEncoder
from .additive_attention_user_encoder import AdditiveAttentionUserEncoder
from .fastformer_user_encoder import FastformerUserEncoder
from .multi_head_self_attention_user_encoder import MultiHeadSelfAttentionUserEncoder
from .transformer_user_encoder import TransformerUserEncoder
from typing import Dict


class HistoryUserEncoderBuilder:
    @classmethod
    def build_history_user_encoder(cls, config: Dict) -> BaseHistoryUserEncoder:
        out_features: int = config["hidden_dim"]
        
        if config["history_user_encoder_method"] == "AdditiveAttention":
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            return AdditiveAttentionUserEncoder(out_features=out_features, query_vector_dim=query_vector_dim)
        
        elif config["history_user_encoder_method"] == "Fastformer":
            num_heads: int = config["history_user_encoder_fastformer_num_heads"]
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            return FastformerUserEncoder(out_features=out_features, num_heads=num_heads, query_vector_dim=query_vector_dim)
        
        elif config["history_user_encoder_method"] == "MultiHeadSelfAttention":
            num_heads: int = config["history_user_encoder_multi_head_self_attention_num_heads"]
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            return MultiHeadSelfAttentionUserEncoder(out_features=out_features, num_heads=num_heads, query_vector_dim=query_vector_dim)
        
        elif config["history_user_encoder_method"] == "Transformer":
            num_heads: int = config["history_user_encoder_transformer_num_heads"]
            query_vector_dim: int = config["additive_attention_query_vector_dim"]
            return TransformerUserEncoder(out_features=out_features, num_heads=num_heads, query_vector_dim=query_vector_dim)
        
        else:
            raise ValueError(f"Invalid history_user_encoder_method, got: {config['history_user_encoder_method']}")