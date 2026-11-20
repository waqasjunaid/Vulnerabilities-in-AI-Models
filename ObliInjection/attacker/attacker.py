import random
import numpy as np
from tqdm import tqdm
from task import InjectedTask, TargetTask
from utils import open_config

from itertools import chain, product, repeat

class Attacker:
    def __init__(self, target_dataset, injected_dataset, attack_permute_num, **kwargs) -> None:
        self.target_dataset = target_dataset
        self.injected_dataset = injected_dataset
        self.attack_permute_num = attack_permute_num
        
        self.str2print = kwargs.get('str2print', None)
        self.keyword = kwargs.get('keyword', None)
        self.x_form = kwargs.get('x_form', None)
        self.question_pos = kwargs.get('question_pos', None)
        
        target_task_config = open_config(f"configs/task_configs/{self.target_dataset}_config.json")
        injected_task_config = open_config(f"configs/task_configs/{self.injected_dataset}_config.json")
        self.target_task = TargetTask(target_task_config, question_pos=self.question_pos)

        self.injected_task = InjectedTask(injected_task_config, print_kwargs={"str2print": self.str2print if self.str2print is not None else injected_task_config["dataset_info"]["str2print"]}, keyword=injected_task_config["dataset_info"]["keyword"] if self.keyword is None else self.keyword)

        self.initial_response = {
            "amazon_reviews": {
                "print": {
                    "llama3": "{text}",
                    "llama4": "{text}",
                    "llama3.1": "{text}",
                    "mistral": "{text}",
                    "Qwen": "{text}",
                    "Qwen3-4B": "{text}",
                    "falcon3": "{text}",
                },
            },
            "hotpotqa": {
                "print": {
                    "llama3": "{text}",
                    "llama4": "{text}",
                    "llama3.1": "{text}",
                    "mistral": "{text}",
                    "Qwen": "{text}",
                    "Qwen3-4B": "{text}",
                    "falcon3": "{text}",
                },
            },
            "multi_news" :{
                "print": {
                    "llama3": "{text}",
                    "llama4": "{text}",
                    "llama3.1": "{text}",
                    "mistral": "{text}",
                    "Qwen": "{text}",
                    "Qwen3-4B": "{text}",
                    "falcon3": "{text}",
                },
            },
        }[self.target_dataset][self.injected_dataset]
    
    def get_attack_indices(self, data, num_permutations):
        source_permutations = [np.random.permutation(len(data[i//num_permutations])) for i in range(len(data)*num_permutations)]
            
        inject_pos = [np.random.randint(0, len(source_permutations[i])) for i in range(len(source_permutations))]
        
        return source_permutations, inject_pos
    
    def construct_armed_payload_with_prefix_postfix(self, data, prefix, postfix, shadow_segments=None):
        inject_ins = self.injected_task.get_injected_instruction()

        for i, inject_data in enumerate(data):
            data[i] = inject_data.rstrip()
            if data[i][-1] not in ['.', '!', '?']: 
                data[i] += '.'
                
        if self.x_form is None or self.x_form == 'prefix_postfix':
            armed_payloads = [f"{prefix} {inject_ins} {inject_data}\n{postfix}" for inject_data in data]
            
        elif self.x_form in ['segment']:
            armed_payloads = [f"{prefix}\n{postfix}" for inject_data in data]
                
        elif self.x_form == 'segment_plus':
            if shadow_segments is not None:
                shadow_segments = list(chain.from_iterable(repeat(segment, len(armed_payloads) // len(shadow_segments)) for segment in shadow_segments))
                assert len(shadow_segments) == len(armed_payloads), "Shadow segments length must match armed payloads length."
            else:
                shadow_segment = "Great Product!"
            armed_payloads = [f"{shadow_segment}{prefix} {inject_ins} {inject_data}\n{postfix}" for inject_data in data]
        else:
            raise NotImplementedError(f"x_form {self.x_form} is not implemented.")
        
        return armed_payloads
    
    def construct_armed_payload_with_advtok(self, data, prefix_len, postfix_len, adv_tok):
        inject_ins = self.injected_task.get_injected_instruction()

        for i, inject_data in enumerate(data):
            data[i] = inject_data.rstrip()
            if data[i][-1] not in ['.', '!', '?']: 
                data[i] += '.'
        armed_payloads = [f"{adv_tok*prefix_len} {inject_ins} {inject_data}\n{adv_tok*postfix_len}" for inject_data in data]
        
        return armed_payloads
    
    def construct_attack_messages(self, sources_lst, armed_payloads, instruction, source_permutations, inject_pos_lst, question_lst):
        attack_messages = []

        step = len(source_permutations) // len(sources_lst)
        for idx, permutation in enumerate(source_permutations):
            
            sources = sources_lst[idx//step]
            inject_pos = inject_pos_lst[idx]

            injected_sources = list(sources) + [armed_payloads[idx//step]]
            injected_indices = list(permutation[:inject_pos]) + [len(sources)] + list(permutation[inject_pos:])

            reordered_sources = [injected_sources[i] for i in injected_indices]
            
            attack_input = self.target_task.get_formatted_input([reordered_sources], instruction,add_instruction=True, question_lst=question_lst)[0]    
            attack_messages.append({'role':'user', 'content': attack_input})

        return attack_messages
            
    def get_attack_ans(self, model, attack_inputs):
        answers = [model.query(attack_data) for attack_data in tqdm(attack_inputs)]
        return answers
        
        
        
        
        
        