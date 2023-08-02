from models.recommend_model import RecommendModel
import torch
from torch import Tensor
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import gc
from sklearn.metrics import roc_auc_score


class RecommendEvaluator:
    def __init__(self, post_id_to_info: Dict[int, Dict], behaviours: List[Dict], num_categories: int, history_length: int,
                 use_pre_encoded_text: bool, use_side_feature: bool):
        self.post_id_to_info: Dict[int, Dict] = post_id_to_info
        self.behaviours: List[Dict] = behaviours
        self.num_categories: int = num_categories
        self.history_length: int = history_length
        self.use_pre_encoded_text: bool = use_pre_encoded_text
        self.use_side_feature: bool = use_side_feature
        self.all_post_ids: List[int] = list(self.post_id_to_info.keys())
        self.post_id_to_encode: Dict[np.ndarray] = {}
        
    def reset(self):
        self.post_id_to_encode.clear()
        gc.collect()
        
    def convert_data_dict_to_device(self, data_dict: Dict, device: torch.device) -> Dict:
        for key, value in data_dict.items():
            if isinstance(value, dict):
                data_dict[key] = self.convert_data_dict_to_device(data_dict=value, device=device)
            elif isinstance(value, Tensor):
                data_dict[key] = value.to(device)
            else:
                raise ValueError("Invalid data type")
        return data_dict
        
    def get_list_posts_features(self, post_ids: List[int]) -> Dict[str, Tensor]:
        def _convert_numpy_to_tensor(_data: np.ndarray) -> Tensor:
            return torch.tensor(_data, dtype=torch.float)
        
        result: Dict[str, Tensor] = {}
        
        posts_info: List[Dict] = list(map(lambda x: self.post_id_to_info[x], post_ids))
        
        if self.use_pre_encoded_text:
            titles: List[np.ndarray] = list(map(lambda x: x["title_bert_encode"], posts_info))
            titles: List[Tensor] = list(map(_convert_numpy_to_tensor, titles))
            title: Tensor = torch.stack(titles, dim=0)
            result["title_pre_encode"] = title

            abstracts: List[np.ndarray] = list(map(lambda x: x["abstract_bert_encode"], posts_info))
            abstracts: List[Tensor] = list(map(_convert_numpy_to_tensor, abstracts))
            abstract: Tensor = torch.stack(abstracts, dim=0)
            result["abstract_pre_encode"] = abstract
        
        else:
            title_token_ids: List[List[int]] = list(map(lambda x: x["title_token_ids"], posts_info))
            title_token_ids: Tensor = torch.tensor(title_token_ids, dtype=torch.int)
            title_attention_mask: Tensor = (title_token_ids != 0).float()
            result["title_token_ids"] = title_token_ids
            result["title_attention_mask"] = title_attention_mask
            
            abstract_token_ids: List[List[int]] = list(map(lambda x: x["abstract_token_ids"], posts_info))
            abstract_token_ids: Tensor = torch.tensor(abstract_token_ids, dtype=torch.int)
            abstract_attention_mask: Tensor = (abstract_token_ids != 0).float()
            result["abstract_token_ids"] = abstract_token_ids
            result["abstract_attention_mask"] = abstract_attention_mask
        
        categories: List[int] = list(map(lambda x: x["category_id"], posts_info))
        category: Tensor = torch.tensor(categories, dtype=torch.int)
        result["category"] = category
        
        subcategories: List[int] = list(map(lambda x: x["subcategory_id"], posts_info))
        subcategory: Tensor = torch.tensor(subcategories, dtype=torch.int)
        result["subcategory"] = subcategory
        
        return result
    
    def update_batch_posts_encodes(self, post_ids: List[int], model: RecommendModel, device: torch.device):
        posts_features: Dict[str, Tensor] = self.get_list_posts_features(post_ids=post_ids)
        posts_features: Dict[str, Tensor] = self.convert_data_dict_to_device(data_dict=posts_features, device=device)
        with torch.no_grad():
            model.eval()
            posts_encodes: Tensor = model.compute_batch_posts_encodes(data=posts_features)
            posts_encodes_numpy: np.ndarray = posts_encodes.cpu().detach().numpy()
        for index, post_id in enumerate(post_ids):
            self.post_id_to_encode[post_id] = posts_encodes_numpy[index]
        del posts_features, posts_encodes
        torch.cuda.empty_cache()
        
    def update_all_posts_encodes(self, model: RecommendModel, device: torch.device):
        batch_size: int = 512
        progress_bar = tqdm(range(0, len(self.all_post_ids), batch_size), "Updating posts encodes....", position=0, leave=True)
        for start_index in progress_bar:
            end_index: int = min(start_index+batch_size, len(self.all_post_ids))
            self.update_batch_posts_encodes(post_ids=self.all_post_ids[start_index:end_index], model=model, device=device)
        progress_bar.close()
        
    def get_batch_posts_encodes(self, post_ids: List[int], max_length: int = None) -> Tensor:
        def _convert_numpy_to_tensor(_data: np.ndarray) -> Tensor:
            return torch.tensor(_data, dtype=torch.float)
        
        if max_length is not None:
            post_ids: List[int] = post_ids + list(map(lambda x: 0, range(max_length - len(post_ids))))
            assert len(post_ids) == max_length
        
        posts_encodes: List[np.ndarray] = list(map(lambda x: self.post_id_to_encode[x], post_ids))
        posts_encodes: List[Tensor] = list(map(_convert_numpy_to_tensor, posts_encodes))
        return torch.stack(posts_encodes, dim=0)
    
    def get_padded_history(self, history: List[int]) -> List[int]:
        history: List[int] = history[:self.history_length]
        history: List[int] = history + list(map(lambda x: 0, range(self.history_length - len(history))))
        assert len(history) == self.history_length
        return history
    
    def get_user_side_feature(self, history: List[int]) -> Tensor:
        side_feature: np.ndarray = np.ones(self.num_categories-1)
        count: int = 0
        for post_id in history:
            info: Dict = self.post_id_to_info[post_id]
            category_id: int = info["category_id"]
            subcategory_id: int = info["subcategory_id"]
            if category_id != 0 or subcategory_id != 0:
                count += 1
                if category_id != 0:
                    side_feature[category_id-1] = side_feature[category_id-1] + 1.0
                if subcategory_id != 0:
                    side_feature[subcategory_id-1] = side_feature[subcategory_id-1] + 1.0
        if count > 0:
            side_feature = side_feature / count
        return torch.tensor(side_feature, dtype=torch.float)
    
    def compute_batch_users_history_encodes(self, histories: List[List[int]]) -> Tuple[Tensor, Tensor]:
        def _get_history_attn_mask(_history: List[int]) -> Tensor:
            return torch.tensor(list(map(lambda x: 0 if x==0 else 1, _history)), dtype=torch.float)
            
        histories: List[List[int]] = list(map(self.get_padded_history, histories))
        histories_encodes: List[Tensor] = list(map(lambda x: self.get_batch_posts_encodes(post_ids=x, max_length=None), histories))
        histories_attn_masks: List[Tensor] = list(map(_get_history_attn_mask, histories))
        return torch.stack(histories_encodes, dim=0), torch.stack(histories_attn_masks, dim=0)
    
    def compute_batch_users_side_features(self, histories: List[List[int]]) -> Tensor:
        side_features: List[Tensor] = list(map(self.get_user_side_feature, histories))
        return torch.stack(side_features, dim=0)
        
    def compute_batch_users_encodes(self, behaviours: List[Dict], model: RecommendModel, device: torch.device) -> Tensor:
        histories: List[List[int]] = list(map(lambda x: x["history"], behaviours))
        history, history_attn_mask = self.compute_batch_users_history_encodes(histories=histories)
        if self.use_side_feature:
            side_feature: Tensor = self.compute_batch_users_side_features(histories=histories)
        else:
            side_feature: Tensor = None
        with torch.no_grad():
            model.eval()
            if self.use_side_feature:
                users_encodes: Tensor = model.compute_batch_users_encodes(history=history.to(device), history_attn_mask=history_attn_mask.to(device), 
                                                                          side_feature=side_feature.to(device))
            else:
                users_encodes: Tensor = model.compute_batch_users_encodes(history=history.to(device), history_attn_mask=history_attn_mask.to(device))
        del history, history_attn_mask, side_feature
        torch.cuda.empty_cache()
        return users_encodes
    
    def get_batch_candidates_encodes(self, behaviours: List[Dict]) -> Tensor:
        candidates: List[List[int]] = list(map(lambda x: x["positive"] + x["negative"], behaviours))
        max_length: int = max(map(lambda x: len(x), candidates))
        candidates_encodes: List[Tensor] = list(map(lambda x: self.get_batch_posts_encodes(post_ids=x, max_length=max_length), candidates))
        return torch.stack(candidates_encodes, dim=0)
    
    def compute_batch_behaviours_scores(self, behaviours: List[Dict], model: RecommendModel, device: torch.device) -> np.ndarray:
        candidate_encode: Tensor = self.get_batch_candidates_encodes(behaviours=behaviours).to(device)
        user_encode: Tensor = self.compute_batch_users_encodes(behaviours=behaviours, model=model, device=device)
        with torch.no_grad():
            model.eval()
            scores: Tensor = model.compute_batch_scores(user_encode=user_encode, candidate_encode=candidate_encode)
            scores_numpy: np.ndarray = scores.cpu().detach().numpy()
        del candidate_encode, user_encode, scores
        torch.cuda.empty_cache()
        return scores_numpy
    
    def auc_score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return roc_auc_score(y_true=y_true, y_score=y_score)

    def mrr_score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order)
        rr_score = y_true / (np.arange(len(y_true)) + 1)
        return float(np.sum(rr_score) / np.sum(y_true))

    def dcg_score(self, y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return float(np.sum(gains / discounts))

    def ndcg_score(self, y_true: np.ndarray, y_score: np.ndarray, k: int):
        best: float = self.dcg_score(y_true=y_true, y_score=y_true, k=k)
        actual: float = self.dcg_score(y_true=y_true, y_score=y_score, k=k)
        return float(actual / best)
    
    def compute_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        result: Dict[str, float] = {
            "AUC": self.auc_score(y_true=y_true, y_score=y_score),
            "MRR": self.mrr_score(y_true=y_true, y_score=y_score),
            "NDCG@3": self.ndcg_score(y_true=y_true, y_score=y_score, k=3),
            "NDCG@5": self.ndcg_score(y_true=y_true, y_score=y_score, k=5),
            "NDCG@10": self.ndcg_score(y_true=y_true, y_score=y_score, k=10)
        }
        return result
    
    def compute_batch_behaviours_metrics(self, behaviours: List[Dict], model: RecommendModel, device: torch.device) -> Dict[str, List[float]]:
        scores: np.ndarray = self.compute_batch_behaviours_scores(behaviours=behaviours, model=model, device=device)
        
        result: Dict[str, List[float]] = {
            "AUC": [],
            "MRR": [],
            "NDCG@3": [],
            "NDCG@5": [],
            "NDCG@10": []
        }
        
        for index, behaviour in enumerate(behaviours):
            labels: List[float] = list(map(lambda x: 1.0, behaviour["positive"])) + list(map(lambda x: 0.0, behaviour["negative"]))
            y_true: np.ndarray = np.array(labels)
            y_score: np.ndarray = scores[index]
            y_score: np.adarray = y_score[:len(labels)]
            instance_result: Dict = self.compute_metrics(y_true=y_true, y_score=y_score)
            for key in result:
                result[key].append(instance_result[key])

        return result
    
    def compute_all_behaviours_metrics(self, model: RecommendModel, device: torch.device) -> Dict[float, List[float]]:
        result: Dict[str, List[float]] = {
            "AUC": [],
            "MRR": [],
            "NDCG@3": [],
            "NDCG@5": [],
            "NDCG@10": []
        }
        
        batch_size: int = 128
        progress_bar = tqdm(range(0, len(self.behaviours), batch_size), "Computing metrics...", position=0, leave=True)
        for start_index in progress_bar:
            end_index: int = min(start_index+batch_size, len(self.behaviours))
            batch_result: Dict[str, List[float]] = self.compute_batch_behaviours_metrics(behaviours=self.behaviours[start_index:end_index], model=model, device=device)
            for key in result:
                result[key] = result[key] + batch_result[key]
        progress_bar.close()
        
        for value in result.values():
            assert len(value) == len(self.behaviours)
        
        return result

    def evaluate(self, model: RecommendModel, device: torch.device) -> Dict[float, List[float]]:
        self.reset()
        self.update_all_posts_encodes(model=model, device=device)
        result: Dict[float, List[float]] = self.compute_all_behaviours_metrics(model=model, device=device)
        self.reset()
        return result