from .base_text_encoder import BaseTextEncoder
from .raw_text_encoder import BaseRawTextEncoder, RawTextEncoderBuilder
from .pre_encoded_text_encoder import BasePreEncodedTextEncoder, PreEncodedTextEncoderBuilder
from typing import Dict
import torch.nn as nn


class TextEncoderBuilder:
    @classmethod
    def build_text_encoder(cls, config: Dict, **kwargs) -> BaseTextEncoder:
        out_features: int = config["hidden_dim"]
        
        if config["text_encoder_method"] == "Raw":
            word_embedding: nn.Embedding = kwargs["word_embedding"]
            text_encoder: BaseRawTextEncoder = RawTextEncoderBuilder.build_raw_text_encoder(config=config, word_embedding=word_embedding)
            return text_encoder
        
        elif config["text_encoder_method"] == "PreEncoded":
            text_encoder: BasePreEncodedTextEncoder = PreEncodedTextEncoderBuilder.build_pre_encoded_text_encoder(config=config)
            return text_encoder
            
        else:
            raise ValueError(f"Invalid text_encoder_method, got: {config['text_encoder_method']}")
