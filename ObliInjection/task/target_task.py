import os
import json
import pandas as pd
import numpy as np

from task.task import Task, TARGET_DATASETS

class TargetTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.dataset_name in TARGET_DATASETS, f"Invalid dataset name: {self.dataset_name}"
        
        self.load_raw_data_func = {
                'amazon_reviews': self.__get_amazon_reviews,
                "multi_news": self.__get_multi_news,
                "hotpotqa": self.__get_hotpotqa,
            }[self.dataset_name]

        self.answer_hint = {
            "amazon_reviews": "Answer: ",
            "multi_news": "Answer: ",
            "hotpotqa": "Answer: ",
        }[self.dataset_name]
        
        self.target_instruction = self.instruction['target']
        
        self.data = self.__prepare_data()
        
    def get_formatted_input(self, sources_lst,instruction, target=None, return_dict=False, add_instruction=False, question_lst=None): 
        formatted_input_lst = []
        
        for idx, sources in enumerate(sources_lst):
            if type(sources) == list:
                question_str = f"Question: {question_lst[idx]}" if question_lst is not None else ""
                
                formatted_input =  question_str + self.source_sep + self.source_sep.join(f"{source}" for i, source in enumerate(sources))+f"\n\n\n\n{self.answer_hint}" 
                
                if add_instruction:
                    formatted_input = f"{instruction}\n\n\n" + formatted_input 
      
                if target:
                    formatted_input += target
                    
                if return_dict:
                    formatted_input_lst.append({
                        'role': 'user',
                        'content': formatted_input,
                    })
                else:
                    formatted_input_lst.append(formatted_input)
            else:
                raise ValueError(f"Invalid input type: {type(sources)}")
            
        return formatted_input_lst
    
    def wrap_armed_payload(self, armed_payloads):
        return [payload for i, payload in enumerate(armed_payloads)]

    def get_shadow_segments(self, target_data_indices):
        target_val_data, _,_ = self.get_data('val','target', data_indices=target_data_indices)
        segments_length = np.array([[len(seg) for seg in sources] for sources in target_val_data])
        topk_indices = np.argsort(segments_length, axis=1)[:, -10:][:, ::-1]
        N = topk_indices.shape[0]
        random_choices = np.random.randint(0, 10, size=N)
        
        chosen_indices = topk_indices[np.arange(N), random_choices]
        shadow_segments = [target_val_data[i][chosen_indices[i]] for i in range(N)]
        return shadow_segments

    def __prepare_data(self):
        
        data_save_dir = self.get_data_saving_path()
        
        for _, dir_path in data_save_dir.items():
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            train_target_data = self._load_processed_dataset(data_save_dir['train_path'], "target")
            train_target_label = self._load_processed_label(data_save_dir['train_path'], "target")
            train_target_question = self._load_processed_question(data_save_dir['train_path'], "target") if self.dataset_name in ['hotpotqa'] else None
            
            test_target_data = self._load_processed_dataset(data_save_dir['test_path'], "target")
            test_target_label = self._load_processed_label(data_save_dir['test_path'], "target")
            test_target_question = self._load_processed_question(data_save_dir['test_path'], "target") if self.dataset_name in ['hotpotqa'] else None
            
            val_target_data = self._load_processed_dataset(data_save_dir['val_path'], "target")
            val_target_label = self._load_processed_label(data_save_dir['val_path'], "target")
            val_target_question = self._load_processed_question(data_save_dir['val_path'], "target") if self.dataset_name in ['hotpotqa'] else None
        except:
            print(f"No data found for {self.dataset_name}. Try to load and process the data from the beginning")
        
            train_target_df, test_target_df, val_target_df = self.load_raw_data_func()
    
            train_target_data = list(train_target_df['data'])
            train_target_label = list(train_target_df['label'])
            train_target_question = list(train_target_df['question']) if self.dataset_name in ['hotpotqa'] else None
                
            np.savez(f"{data_save_dir['train_path']}/target_data.npz", data=train_target_data, label=train_target_label, question=train_target_question)
            
            test_target_data = list(test_target_df['data'])
            test_target_label = list(test_target_df['label'])
            test_target_question = list(test_target_df['question']) if self.dataset_name in ['hotpotqa'] else None

            np.savez(f"{data_save_dir['test_path']}/target_data.npz", data=test_target_data, label=test_target_label, question=test_target_question)
            
            val_target_data = list(val_target_df['data'])
            val_target_label = list(val_target_df['label'])
            val_target_question = list(val_target_df['question']) if self.dataset_name in ['hotpotqa'] else None
            np.savez(f"{data_save_dir['val_path']}/target_data.npz", data=val_target_data, label=val_target_label, question=val_target_question)
            
        return {
            'train': {
                'data': train_target_data,
                'label': train_target_label,
                'question': train_target_question},
            'test': {
                'data': test_target_data,
                'label': test_target_label,
                'question': test_target_question},
            'val': {
                'data': val_target_data,
                'label': val_target_label,
                'question': val_target_question},
            }
        
    def __get_amazon_reviews(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        
        with open(self.data_path['train'], "r") as train_file:
            train_data = json.load(train_file)
            train_sources = [train_data[i]['sources'] for i in range(len(train_data))]
            
            if isinstance(train_sources[0][0], dict):
                train_sources = [[source['Text'] if not isinstance(source['Text'],list) else '\n'.join(source['Text']) for source in sources] for sources in train_sources]

            try:
                train_label = [train_data[i]['label'] for i in range(len(train_data))]
            except:
                train_label = ["" for i in range(len(train_data))]
        
        with open(self.data_path["test"], "r") as test_file:
            test_data = json.load(test_file)
            test_sources = [test_data[i]['sources'] for i in range(len(test_data))]
            try:
                test_label = [test_data[i]['label'] for i in range(len(test_data))]
            except:
                test_label = ["" for i in range(len(test_data))]
            
            if isinstance(test_sources[0][0], dict):
                test_sources = [[source['Text'] if not isinstance(source['Text'],list) else '\n'.join(source['Text']) for source in sources] for sources in test_sources]
        
        with open(self.data_path["val"], "r") as val_file:
            val_data = json.load(val_file)
            val_sources = [val_data[i]['sources'] for i in range(len(val_data))]
            try:
                val_label = [val_data[i]['label'] for i in range(len(val_data))]
            except:
                val_label = ["" for i in range(len(val_data))]
            
            if isinstance(val_sources[0][0], dict):
                val_sources = [[source['Text'] if not isinstance(source['Text'],list) else '\n'.join(source['Text']) for source in sources] for sources in val_sources]
        
        train_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in train_sources]
        val_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in val_sources]
        test_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in test_sources]

        train_df = pd.DataFrame({
            'data': train_sources,
            'label': train_label
        })
        
        test_df = pd.DataFrame({
            'data': test_sources,
            'label': test_label
        })
        
        val_df = pd.DataFrame({
            'data': val_sources,
            'label': val_label
        })
        return train_df, test_df, val_df
    
    def __get_multi_news(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        
        with open(self.data_path['train'], "r") as train_file:
            train_data = json.load(train_file)
            train_sources = [train_data[i]['sources'] for i in range(len(train_data))]
            
            if isinstance(train_sources[0][0], dict):
                train_sources = [[source['Text'] if not isinstance(source['Text'],list) else '\n'.join(source['Text']) for source in sources] for sources in train_sources]
            try:
                train_label = [train_data[i]['label'] for i in range(len(train_data))]
            except:
                train_label = ["" for i in range(len(train_data))]
        
        with open(self.data_path["test"], "r") as test_file:
            test_data = json.load(test_file)
            test_sources = [test_data[i]['sources'] for i in range(len(test_data))]
            try:
                test_label = [test_data[i]['label'] for i in range(len(test_data))]
            except:
                test_label = ["" for i in range(len(test_data))]
            
            if isinstance(test_sources[0][0], dict):
                test_sources = [[source['Text'] if not isinstance(source['Text'],list) else '\n'.join(source['Text']) for source in sources] for sources in test_sources]
        
        with open(self.data_path["val"], "r") as val_file:
            val_data = json.load(val_file)
            val_sources = [val_data[i]['sources'] for i in range(len(val_data))]
            try:
                val_label = [val_data[i]['label'] for i in range(len(val_data))]
            except:
                val_label = ["" for i in range(len(val_data))]
            
            if isinstance(val_sources[0][0], dict):
                val_sources = [[source['Text'] if not isinstance(source['Text'],list) else '\n'.join(source['Text']) for source in sources] for sources in val_sources]
        
        train_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in train_sources]
        val_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in val_sources]
        test_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in test_sources]

        train_df = pd.DataFrame({
            'data': train_sources,
            'label': train_label
        })
        
        test_df = pd.DataFrame({
            'data': test_sources,
            'label': test_label
        })
        
        val_df = pd.DataFrame({
            'data': val_sources,
            'label': val_label
        })
        return train_df, test_df, val_df
    
    def __get_hotpotqa(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        
        with open(self.data_path['train'], "r") as train_file:
            train_data = json.load(train_file)
            train_sources = [train_data[i]['sources'] for i in range(len(train_data))]
            train_question = [train_data[i]['question'] for i in range(len(train_data))]
            
            if isinstance(train_sources[0][0], dict):
                train_sources = [[f"{source['Title']}\n{source['Text']}" if not isinstance(source['Text'],list) else source['Title']+'\n'+'\n'.join(source['Text']) for source in sources] for sources in train_sources]
            try:
                train_label = [train_data[i]['label'] for i in range(len(train_data))]
            except:
                train_label = ["" for i in range(len(train_data))]
        
        with open(self.data_path["test"], "r") as test_file:
            test_data = json.load(test_file)
            test_sources = [test_data[i]['sources'] for i in range(len(test_data))]
            try:
                test_label = [test_data[i]['label'] for i in range(len(test_data))]
            except:
                test_label = ["" for i in range(len(test_data))]
            test_question = [test_data[i]['question'] for i in range(len(test_data))]
            
            if isinstance(test_sources[0][0], dict):
                test_sources = [[f"{source['Title']}\n{source['Text']}" if not isinstance(source['Text'],list) else source['Title']+'\n'+'\n'.join(source['Text']) for source in sources] for sources in test_sources]
        
        with open(self.data_path["val"], "r") as val_file:
            val_data = json.load(val_file)
            val_sources = [val_data[i]['sources'] for i in range(len(val_data))]
            try:
                val_label = [val_data[i]['label'] for i in range(len(val_data))]
            except:
                val_label = ["" for i in range(len(val_data))]
            val_question = [val_data[i]['question'] for i in range(len(val_data))]
            
            if isinstance(val_sources[0][0], dict):
                val_sources = [[f"{source['Title']}\n{source['Text']}" if not isinstance(source['Text'],list) else source['Title']+'\n'+'\n'.join(source['Text']) for source in sources] for sources in val_sources]
        
        train_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in train_sources]
        val_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in val_sources]
        test_sources = [[source + '.' if not source.endswith(('.', '!', '?')) else source for source in sources ] for sources in test_sources]

        train_df = pd.DataFrame({
            'data': train_sources,
            'label': train_label,
            'question': train_question
        })
        
        test_df = pd.DataFrame({
            'data': test_sources,
            'label': test_label,
            'question': test_question
        })
        
        val_df = pd.DataFrame({
            'data': val_sources,
            'label': val_label,
            'question': val_question
        })
        return train_df, test_df, val_df
