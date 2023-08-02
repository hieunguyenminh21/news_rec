from .base_user_encoder import BaseUserEncoder
from .history_user_encoder import BaseHistoryUserEncoder, HistoryUserEncoderBuilder
from typing import Dict


class UserEncoderBuilder:
    @classmethod
    def build_user_encoder(cls, config: Dict):
        if config["user_encoder_method"] == "History":
            user_encoder: BaseHistoryUserEncoder = HistoryUserEncoderBuilder.build_history_user_encoder(config=config)
            return user_encoder
        else:
            raise ValueError(f"Invalid user_encoder_method, got: {config['user_encoder_method']}")
