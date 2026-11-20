import os
import json
import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_DATASETS = ['amazon_reviews', "multi_news", "hotpotqa"]
INJECT_DATASETS = ['print']

class Task:
    def __init__(self, config, **kwargs) -> None:
        self.task = config["task_info"]["task"]
        self.instruction = config["task_info"]["instruction"]
        self.task_type = config["task_info"]["type"]
        self.source_sep = config["task_info"]["source_sep"]
        self.dataset_name = config["dataset_info"]["dataset_name"]
        self.data_path = config["dataset_info"]["data_path"]
        self.source_num = config["dataset_info"]["source_num"]
        
        self.metric = {
            'print': 'accuracy',
            'amazon_reviews': 'rouge',
            "multi_news": 'rouge',
            "hotpotqa": 'rouge',
        }[self.dataset_name]
            
    def _load_processed_dataset(self, save_dir, target_injected):
        data_path = f'{save_dir}/{target_injected}_data.npz'
        return list(np.load(f'{data_path}')['data'])
    
    def _load_processed_label(self, save_dir, target_injected):
        data_path = f'{save_dir}/{target_injected}_data.npz'
        return list(np.load(f'{data_path}')['label'])
    
    def _load_processed_question(self, save_dir, target_injected):
        data_path = f'{save_dir}/{target_injected}_data.npz'
        return list(np.load(f'{data_path}')['question'])
    
    def get_data_saving_path(self):
        saving_path = f"./data/{self.dataset_name}"
        train_path = f"{saving_path}/train/source_num_{self.source_num['train']}"
        test_path = f"{saving_path}/test/source_num_{self.source_num['test']}"
        val_path = f"{saving_path}/val/source_num_{self.source_num['val']}"
        
        return {'path': saving_path, 'train_path': train_path, 'test_path': test_path, 'val_path': val_path}
    
    def get_target_instruction(self):
        return self.tareget_instruction

    def get_injected_instruction(self):
        return self.injected_instruction
    
    def get_metric(self):
        return self.metric
    
    def get_data(self, train_val_test, target_injected, num=None, data_indices=None):
        if target_injected in ['injected'] and self.dataset_name in ['amazon_reviews']:
            raise ValueError(f"{self.dataset_name} do not have injected dataset")
        
        requested_data = self.data[train_val_test]['data']
        requested_label = self.data[train_val_test]['label']
        if self.dataset_name in ['hotpotqa']:
            requested_question = self.data[train_val_test]['question']
        else:
            requested_question = None

        if data_indices:
            selected_data = [requested_data[i] for i in data_indices]
            selected_label = [requested_label[i] for i in data_indices]
            if self.dataset_name in ['hotpotqa']:
                selected_question = [requested_question[i] for i in data_indices]
            else: 
                selected_question = None

            return selected_data, selected_label, selected_question 
        
        if num is None:
            num = len(requested_data)
            
        assert len(requested_data) == len(requested_label)
        if len(requested_data) >= num:
            if self.dataset_name in ['hotpotqa']:
                return requested_data[:num], requested_label[:num], requested_question[:num]
            else:
                return requested_data[:num], requested_label[:num], None
        else:
            print(f"[{self.dataset} | {train_test_split} | {target_injected}] has a length of {len(requested_data)} which is smaller than {num}")
            if self.dataset_name in ['hotpotqa']:
                return requested_data, requested_label, requested_question
            else:
                return requested_data, requested_label,None
        
    def convert_label_to_string(self, labels):
        if self.dataset_name == 'print':
            return [self.keyword if label==1 else f"not {self.keyword}" for label in labels]
        else:
            raise ValueError(f"{self.dataset} is not supported")