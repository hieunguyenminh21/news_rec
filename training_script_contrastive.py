from common.logger import SingletonLogger
from common.utils import LocalFileHandlerUtils, PickleReadObjectFromLocalPatient, JsonWriteObjectToLocalPatient
from datasets import RecommendDataset, RecommendDatasetBuilder, RecommendEvaluator, RecommendEvaluatorBuilder
from models.recommend_model import RecommendModel, RecommendModelBuilder
from models.losses import BPRLoss, ContrastiveLoss

import torch
import random
import numpy as np
import pandas as pd
import pynvml

from time import sleep
from typing import Dict
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

SingletonLogger.set_logger_name("Train recommend model")

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

all_datasets = ['kenh14', 'soha']


i_u_encoders = ['CNN-AdditiveAttention', 'CNN-Fastformer', 'CNN-MultiHeadSelfAttention', 'CNN-Transformer', \
                'Fastformer-AdditiveAttention', 'Fastformer-Fastformer', 'Fastformer-MultiHeadSelfAttention', 'Fastformer-Transformer', \
                'MultiHeadSelfAttention-AdditiveAttention', 'MultiHeadSelfAttention-Fastformer', \
                'MultiHeadSelfAttention-MultiHeadSelfAttention', 'MultiHeadSelfAttention-Transformer', \
                'Transformer-AdditiveAttention', 'Transformer-Fastformer', 'Transformer-MultiHeadSelfAttention', 'Transformer-Transformer']

# ---------------------------------------- CHECKER ---------------------------------------- #
df = pd.read_csv('./csv_result.csv')
df = df[['data_dir', 'text_encoder_method' ,'raw_text_encoder_method', 'history_user_encoder_method', 'user_history_augmentation_method', 'recommend_weight']]

raw_text_encoder_method = []
history_user_encoder_method = []
recommend_weight = []
combines_trained = []

for index, row in df.iterrows():
    d, t, r, h, u, w = row
    d = d.split('/')[2]
    combines_trained.append(d + '-' + t + '-' + r + '-' + h + '-' + u + '-' + str(w))

    
def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

cuda_device = '1'

while True:
    if get_memory_free_MiB(cuda_device) > 11000:
        break
    else:
        print("het mem")
        sleep(300)


# ---------------------------------------- MAIN ---------------------------------------- #
for i_u in i_u_encoders:
    i_u = i_u.split('-')
    print(i_u)
    for dataset in all_datasets:
        for recommend_weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] : # test random state
            # -------------------------------------------- CONFIG -------------------------------------------- #
            DATA_DIR: str = f"data/version_1/{dataset}"
            # LOG_DIR: str = f"{DATA_DIR}/log_dir/recommend_contrastive/{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}"
            LOG_DIR: str = f"{DATA_DIR}/log_dir/recommend/new/{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}"
            LocalFileHandlerUtils.check_and_make_directory(LOG_DIR)
            print(f"ALL RESULTS SAVE IN {LOG_DIR}")
            CATEGORY_TO_INT: Dict[str, int] = PickleReadObjectFromLocalPatient().read(file_name=f"{DATA_DIR}/object_to_int/category_to_int.pkl")
            NUM_CATEGORIES: int = len(CATEGORY_TO_INT)
            VECTORIZER: CountVectorizer = PickleReadObjectFromLocalPatient().read(file_name=f"{DATA_DIR}/vectorizer/vectorizer.pkl")
            NUM_WORDS: int = len(VECTORIZER.vocabulary_)


            CONFIG: Dict = {
                "data_dir": DATA_DIR,
                
                "hidden_dim": 256,
                
                "post_encoder_method": "TitleAbtractCategorySubcategory",
                "additive_attention_query_vector_dim": 224,
                
                "text_encoder_method": "PreEncoded",
                
                "raw_text_encoder_method": i_u[0],
                "raw_text_encoder_cnn_kernel_size": 3,
                "raw_text_encoder_fastformer_num_heads": 16,
                "raw_text_encoder_multi_head_self_attention_num_heads": 16,
                "raw_text_encoder_transformer_num_heads": 16,
                
                "pre_encoded_text_encoder_method": "MLP",
                "text_pre_encoded_dim": 768,
                
                "category_encoder_method": "MLP",
                "category_embedding_dim": 100,
                "num_categories": NUM_CATEGORIES,
                
                "user_encoder_method": "History",
                
                "history_user_encoder_method": i_u[1],
                "history_user_encoder_fastformer_num_heads": 16,
                "history_user_encoder_multi_head_self_attention_num_heads": 16,
                "history_user_encoder_transformer_num_heads": 16,
                
                "history_length": 30,
                "num_negative_samples": 4,
                
                "use_contrastive_learning": True,
                "crop_rate": 0.6,
                "insert_rate": 0.5,
                "mask_rate": 0.3,
                "reorder_rate": 0.3,
                "substitute_rate": 0.3,
                "history_length_threshold": 10,
                "user_history_augmentation_method": "Version2",
                
                "batch_size": 128,
                "recommend_learning_rate": 0.0001,
                "recommend_weight": recommend_weight,
                "contrastive_weight": 1 - recommend_weight,
                "drop_out": 0.2,
                
                "epochs_multi_tasks": 2,
                "num_steps_show_dev": 300,
                
            }
            
            checker = dataset + '-' + CONFIG["text_encoder_method"] + '-' + CONFIG["raw_text_encoder_method"] + '-' + CONFIG["history_user_encoder_method"] + '-' + CONFIG['user_history_augmentation_method'] + '-' + str(CONFIG["recommend_weight"])
            if checker in combines_trained:
                print(f"ALREADY TRAINED {checker}")
                continue
            
            JsonWriteObjectToLocalPatient().write(x=CONFIG, file_name=f"{LOG_DIR}/config.json")

            DEVICE = torch.device(f"cuda:{cuda_device}") if torch.cuda.is_available() else torch.device("cpu")
            # DEVICE = torch.device("cpu")
            print(f"USING DEVICE {DEVICE}")

            # ---------------------------------------- Util Function ---------------------------------------- #
            def print_data_dict_info(data_dict: Dict, prefix: str = ""):
                for key, value in data_dict.items():
                    if isinstance(value, Tensor):
                        print(f"{prefix}{key}: {value.shape}")
                    elif isinstance(value, dict):
                        print(f"{prefix}{key}:")
                        print_data_dict_info(data_dict=value, prefix=prefix+"     ")
                    else:
                        raise ValueError("Invalid data type")
                    
            # ---------------------------------------- Dataset & Evaluator ---------------------------------------- #
            def seed_worker(worker_id):
                worker_seed = 0
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(0)
            
            
            TRAIN_DATASET: RecommendDataset = RecommendDatasetBuilder.build_recommend_dataset(config=CONFIG, data_split_dir="train")
            DEV_EVALUATOR: RecommendEvaluator = RecommendEvaluatorBuilder.build_recommend_evaluator(config=CONFIG, data_split_dir="dev")
            TEST_EVALUATOR: RecommendEvaluator = RecommendEvaluatorBuilder.build_recommend_evaluator(config=CONFIG, data_split_dir="test")
            TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=CONFIG["batch_size"], num_workers=0, worker_init_fn=seed_worker, generator=g, shuffle=True)

            print_data_dict_info(data_dict=next(iter(TRAIN_DATALOADER)))

            # ---------------------------------------- Model ---------------------------------------- #
            RECOMMEND_MODEL: RecommendModel = RecommendModelBuilder.build_recommend_model(config=CONFIG).to(DEVICE)
            print(RECOMMEND_MODEL)

            # ---------------------------------------- Losses ---------------------------------------- #
            RECOMMEND_LOSS_FUNC = BPRLoss().to(DEVICE)
            CONTRASTIVE_LOSS_FUNC = ContrastiveLoss().to(DEVICE)

            # ---------------------------------------- Optimizer ---------------------------------------- #
            OPTIMIZER = Adam([
                {
                    "params": RECOMMEND_MODEL.parameters(),
                    "lr": CONFIG["recommend_learning_rate"]
                },
            ])

            # ---------------------------------------- Train ---------------------------------------- #
            def convert_data_dict_to_device(data_dict: Dict, device: torch.device) -> Dict:
                for key, value in data_dict.items():
                    if isinstance(value, dict):
                        data_dict[key] = convert_data_dict_to_device(data_dict=value, device=device)
                    elif isinstance(value, Tensor):
                        data_dict[key] = value.to(device)
                    else:
                        raise ValueError("Invalid data type")
                return data_dict

            from typing import List, Dict
            import torch.nn as nn


            RECOMMEND_WEIGHT: float = CONFIG["recommend_weight"]
            CONTRASTIVE_WEIGHT: float = 1 - RECOMMEND_WEIGHT


            def multi_tasks_train_step(data_dict: Dict, global_step: int):
                RECOMMEND_MODEL.train()
                
                data_dict: Dict = convert_data_dict_to_device(data_dict=data_dict, device=DEVICE)
                
                ######################### RECOMMEND LOSS ############################
                candidate_encode: Tensor = RECOMMEND_MODEL.compute_batch_list_posts_encodes(data=data_dict["candidate_features"])
                
                history_encode: Tensor = RECOMMEND_MODEL.compute_batch_list_posts_encodes(data=data_dict["history_features"])
                user_encode: Tensor = RECOMMEND_MODEL.compute_batch_users_encodes(history=history_encode, history_attn_mask=data_dict["history_attn_mask"], 
                                                                                side_feature=data_dict.get("side_feature", None))
                
                recommend_score: Tensor = RECOMMEND_MODEL.compute_batch_scores(user_encode=user_encode, candidate_encode=candidate_encode)
                recommend_loss: Tensor = RECOMMEND_LOSS_FUNC(x=recommend_score, target=data_dict["candidate_label"])
                
                ############################# CONTRASTIVE LOSS ############################
                history_encode_1: Tensor = RECOMMEND_MODEL.compute_batch_list_posts_encodes(data=data_dict["history_features_1"])
                user_encode_1: Tensor = RECOMMEND_MODEL.compute_batch_users_encodes(history=history_encode_1, history_attn_mask=data_dict["history_attn_mask_1"], 
                                                                                    side_feature=data_dict.get("side_feature_1", None))
                
                history_encode_2: Tensor = RECOMMEND_MODEL.compute_batch_list_posts_encodes(data=data_dict["history_features_2"])
                user_encode_2: Tensor = RECOMMEND_MODEL.compute_batch_users_encodes(history=history_encode_2, history_attn_mask=data_dict["history_attn_mask_2"], 
                                                                                    side_feature=data_dict.get("side_feature_2", None))
                
                contrastive_loss: Tensor = CONTRASTIVE_LOSS_FUNC(x=user_encode_1, y=user_encode_2)
                
                ################################# COMBINE LOSSES #######################################
                total_loss = RECOMMEND_WEIGHT*recommend_loss + CONTRASTIVE_WEIGHT*contrastive_loss
                
                ########################## BACKWARD ###########################
                OPTIMIZER.zero_grad()
                total_loss.backward()
                OPTIMIZER.step()
                
                ############################### SAVE TRAINING RESULT ################################
                total_loss_val: float = total_loss.item()
                recommend_loss_val: float = recommend_loss.item()
                contrastive_loss_val: float = contrastive_loss.item()
                
                recommend_weight: float = RECOMMEND_WEIGHT
                contrastive_weight: float = CONTRASTIVE_WEIGHT
                
                with open(f"{LOG_DIR}/train_sum_up.txt", mode="a") as file_obj:
                    file_obj.write(f"STEP {global_step}: \n")
                    file_obj.write(f"      total loss {total_loss_val:.4f}      recommend loss {recommend_loss_val:.4f}      contrastive loss {contrastive_loss_val:.4f} \n")
                    file_obj.write(f"      recommend weight {recommend_weight:.4f}      contrastive weight {contrastive_weight:.4f} \n")
                
                del data_dict, candidate_encode, history_encode, user_encode, recommend_score, recommend_loss
                del history_encode_1, user_encode_1, history_encode_2, user_encode_2, contrastive_loss
                del total_loss
                
                torch.cuda.empty_cache()
                
            # ---------------------------------------- Eval ---------------------------------------- #
            from typing import List, Dict
            import numpy as np


            def multi_tasks_dev_step(global_step: int):
                recommend_result: Dict[str, List[float]] = DEV_EVALUATOR.evaluate(model=RECOMMEND_MODEL, device=DEVICE)
                
                mean_auc: float = np.mean(recommend_result["AUC"])
                mean_mrr: float = np.mean(recommend_result["MRR"])
                mean_ndcg_3: float = np.mean(recommend_result["NDCG@3"])
                mean_ndcg_5: float = np.mean(recommend_result["NDCG@5"])
                mean_ndcg_10: float = np.mean(recommend_result["NDCG@10"])
                
                with open(f"{LOG_DIR}/validation_sum_up.txt", mode="a") as file_obj:
                    file_obj.write(f"STEP {global_step}: \n")
                    file_obj.write(f"      AUC {mean_auc:.4f}      MRR {mean_mrr:.4f}     NDCG@3 {mean_ndcg_3:.4f}      NDCG@5 {mean_ndcg_5:.4f}      NDCG@10 {mean_ndcg_10:.4f} \n")
                
                return mean_auc

            # ---------------------------------------- Test ---------------------------------------- #
            from common.utils import PickleWriteObjectToLocalPatient


            def multi_tasks_test_step():
                RECOMMEND_MODEL.load_state_dict(torch.load(f"{LOG_DIR}/recommend_model.pt"))
                
                recommend_result: Dict[str, List[float]] = TEST_EVALUATOR.evaluate(model=RECOMMEND_MODEL, device=DEVICE)
                
                mean_auc: float = np.mean(recommend_result["AUC"])
                mean_mrr: float = np.mean(recommend_result["MRR"])
                mean_ndcg_3: float = np.mean(recommend_result["NDCG@3"])
                mean_ndcg_5: float = np.mean(recommend_result["NDCG@5"])
                mean_ndcg_10: float = np.mean(recommend_result["NDCG@10"])
                
                with open(f"{LOG_DIR}/test_sum_up.txt", mode="a") as file_obj:
                    file_obj.write(f"TEST RESULT: \n")
                    file_obj.write(f"      AUC {mean_auc:.4f}      MRR {mean_mrr:.4f}     NDCG@3 {mean_ndcg_3:.4f}      NDCG@5 {mean_ndcg_5:.4f}      NDCG@10 {mean_ndcg_10:.4f} \n")
                print(f"TEST RESULT: AUC {mean_auc:.4f}      MRR {mean_mrr:.4f}     NDCG@3 {mean_ndcg_3:.4f}      NDCG@5 {mean_ndcg_5:.4f}      NDCG@10 {mean_ndcg_10:.4f} \n")
                PickleWriteObjectToLocalPatient().write(x=recommend_result, file_name=f"{LOG_DIR}/test_instances_metrics.pkl")
                
            # ---------------------------------------- Main Script ---------------------------------------- #
            from tqdm import tqdm


            BEST_AUC: float = 0.0
            GLOBAL_STEP: int = 0


            for epoch in range(CONFIG["epochs_multi_tasks"]):
                
                progress_bar = tqdm(TRAIN_DATALOADER, f"Epoch {epoch}: current best AUC {BEST_AUC:.4f}...", position=0, leave=True)
                for data_dict in progress_bar:
                    multi_tasks_train_step(data_dict=data_dict, global_step=GLOBAL_STEP)
                    
                    if GLOBAL_STEP % CONFIG["num_steps_show_dev"] == 0:
                        step_auc: float = multi_tasks_dev_step(global_step=GLOBAL_STEP)
                        
                        if step_auc > BEST_AUC:
                            BEST_AUC = step_auc
                            torch.save(RECOMMEND_MODEL.state_dict(), f"{LOG_DIR}/recommend_model.pt")
                            progress_bar.set_description(f"Epoch {epoch}: current best AUC {BEST_AUC:.4f}...")
                            progress_bar.update()
                    
                    GLOBAL_STEP += 1
                
                if epoch+1 == CONFIG["epochs_multi_tasks"]:
                    step_auc: float = multi_tasks_dev_step(global_step=GLOBAL_STEP)
                    
                    if step_auc > BEST_AUC:
                        BEST_AUC = step_auc
                        torch.save(RECOMMEND_MODEL.state_dict(), f"{LOG_DIR}/recommend_model.pt")
                        progress_bar.set_description(f"Epoch {epoch}: current best AUC {BEST_AUC:.4f}...")
                        progress_bar.update()
                
                progress_bar.close()
                
            multi_tasks_test_step()