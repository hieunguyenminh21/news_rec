from .recommend_evaluator import RecommendEvaluator
from common.utils import PickleReadObjectFromLocalPatient
from typing import Dict
import gc


class RecommendEvaluatorBuilder:
    @classmethod
    def build_recommend_evaluator(cls, config: Dict, data_split_dir: str, reduce_data: bool = True) -> RecommendEvaluator:
        data_dir: str = config["data_dir"]
        post_id_to_info: Dict[int, Dict] = PickleReadObjectFromLocalPatient().read(file_name=f"{data_dir}/{data_split_dir}/post_id_to_info.pkl")
        behaviours: List[Dict] = PickleReadObjectFromLocalPatient().read(file_name=f"{data_dir}/{data_split_dir}/behaviours.pkl")
        
        if config["text_encoder_method"] == "PreEncoded":
            use_pre_encoded_text: bool = True
        else:
            use_pre_encoded_text: bool = False
            
        if config["user_encoder_method"] == "HistorySideFeature":
            use_side_feature: bool = True
        else:
            use_side_feature: bool = False
        
        if reduce_data:
            for info in post_id_to_info.values():
                if 'title' in info:
                    info.pop("title")
                if 'abstract' in info:
                    info.pop("abstract")
                if 'body' in info:
                    info.pop("body")
                if 'category' in info:
                    info.pop("category")
                if 'subcategory' in info:
                    info.pop("subcategory")

                if use_pre_encoded_text:
                    if "title_token_ids" in info:
                        info.pop("title_token_ids")
                    if "abstract_token_ids" in info:
                        info.pop("abstract_token_ids")
                else:
                    if "title_bert_encode" in info:
                        info.pop("title_bert_encode")
                    if "abstract_bert_encode" in info:
                        info.pop("abstract_bert_encode")

                if "content_bow" in info: 
                    info.pop("content_bow")

            gc.collect()
            
        return RecommendEvaluator(post_id_to_info=post_id_to_info, behaviours=behaviours, 
                                  use_pre_encoded_text=use_pre_encoded_text, use_side_feature=use_side_feature,
                                  num_categories=config['num_categories'], history_length=config["history_length"])
