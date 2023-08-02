from .post_encoder import BasePostEncoder, PostEncoderBuilder
from .user_encoder import BaseUserEncoder, UserEncoderBuilder
from .recommend_model import RecommendModel
from typing import Dict


class RecommendModelBuilder:
    @classmethod
    def build_recommend_model(cls, config: Dict) -> RecommendModel:
        hidden_dim: int = config["hidden_dim"]
        post_encoder: BasePostEncoder = PostEncoderBuilder.build_post_encoder(config=config)
        user_encoder: BaseUserEncoder = UserEncoderBuilder.build_user_encoder(config=config)
        return RecommendModel(out_features=hidden_dim, post_encoder=post_encoder, user_encoder=user_encoder)
