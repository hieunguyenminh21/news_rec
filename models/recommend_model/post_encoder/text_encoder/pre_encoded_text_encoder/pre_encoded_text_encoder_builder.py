from .base_pre_encoded_text_encoder import BasePreEncodedTextEncoder
from .mlp_text_encoder import MLPTextEncoder
from typing import Dict


class PreEncodedTextEncoderBuilder:
    @classmethod
    def build_pre_encoded_text_encoder(cls, config: Dict) -> BasePreEncodedTextEncoder:
        in_features: int = config["text_pre_encoded_dim"]
        out_features: int = config["hidden_dim"]
        
        if config["pre_encoded_text_encoder_method"] == "MLP":
            return MLPTextEncoder(out_features=out_features, in_features=in_features)
        
        else:
            raise ValueError(f"Invalid pre_encoded_text_encoder_method, got: {config['pre_encoded_text_encoder_method']}")
