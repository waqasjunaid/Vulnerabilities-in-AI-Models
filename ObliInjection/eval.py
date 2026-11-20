import os
import csv
import json
import argparse
import torch
import random
import logging
import numpy as np

from task import InjectedTask, TargetTask, Task, TARGET_DATASETS, INJECT_DATASETS
from attacker import Attacker
from dataclasses import fields
from evaluator import EvaluatorTask, EvaluatorAttack
from model import LocalLLM
from utils import open_config, seed_everything, setup_logger
from gcg.gcg import GCG, GCGconfig
from itertools import chain, repeat

def main(args):
    model_config = open_config(f"configs/model_configs/{args.model}_config.json")
    model_config['params']['gpus'] = args.gpus
    model_config['params']['seed'] = args.seed
    model = LocalLLM(model_config)
    
    attacker = Attacker(
        target_dataset=args.target_dataset, 
        injected_dataset=args.injected_dataset, 
        attack_permute_num=args.attack_permute_num, 
        str2print=args.str2print, 
        keyword=args.keyword)
    
    target_shadow_segments = attacker.target_task.get_shadow_segments(args.target_data_indices) if args.use_shadow_segment else None

    evaluator = EvaluatorAttack(args.target_dataset, args.injected_dataset, keyword=attacker.injected_task.keyword)

    target_test_data, _, target_test_question = attacker.target_task.get_data('test','target', data_indices=args.target_data_indices)
    injected_test_data, injected_test_label,_ = attacker.injected_task.get_data('test', 'injected', data_indices=args.injected_data_indices)

    injected_test_label= list(chain.from_iterable(repeat(item, args.target_data_num*args.review_lst_num) for item in injected_test_label))
    injected_test_data= list(chain.from_iterable(repeat(item, args.target_data_num*args.review_lst_num) for item in injected_test_data))

    selected_num_lst = [[args.selected_num for _ in range(args.review_lst_num)]for data in target_test_data]
    
    target_test_data = [random.sample(list(data), k=selected_num_lst[i][j]) for i, data in enumerate(target_test_data) for j in range(len(selected_num_lst[i]))]

    
    for attack in args.optim:
        if args.json_file is not None:
            with open(args.json_file, "r") as f:
                pre_postfix_dic = json.load(f)[attack][model.provider]
        elif os.path.exists(f"attack_results/{args.target_dataset}/{attack}.json"): 
            with open(f"attack_results/{args.target_dataset}/{attack}.json", "r") as f:
                pre_postfix_dic = json.load(f)[attack][model.provider]
        else:
            raise FileNotFoundError("Please provide a json file or make sure the attack results exist.")
        
        gcg_config = open_config(f"configs/attack_configs/gcg_config.json")
        log_dir = f"{args.log_path}_eval/{attack}/{args.model}/{args.target_dataset}_{args.injected_dataset}/" + "target_indices_" + "_".join([str(i) for i in args.target_data_indices])
        
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = f"{log_dir}/seed_{args.seed}.log"
        csv_path = f"{log_dir}/seed_{args.seed}.csv"
        resp_path = f"{log_dir}/seed_{args.seed}_resp.csv"
        
        args.logger = setup_logger(log_path)
        args.logger.info(vars(args))

        for key, value in vars(args).items():
            if key in [field.name for field in fields(GCGconfig)]:
                gcg_config[key] = value
        gcg_config = GCGconfig(**gcg_config)
            
        gcg = GCG(llm=model, config=gcg_config, attacker=attacker, evaluator=evaluator, logger=args.logger)
        
        with open(csv_path, "w") as eval_f:
            writer = csv.writer(eval_f)
            writer.writerow(
                ['seed','attack', "target_data_indice", "prefix", "postfix",'train_loss', 'test_asv']) 
        
        for i, entry in enumerate(pre_postfix_dic):
            test_asv, _, _ = gcg.evaluate_prefix_postfix(
                target_test_data=target_test_data, 
                injected_test_data=injected_test_data,
                injected_test_label=injected_test_label, 
                prefix=entry['prefix'], 
                postfix=entry['postfix'],
                instruction=gcg.attacker.target_task.target_instruction['test'] ,
                target_indice=args.target_data_indices[0], 
                val_test='test', 
                question_lst=target_test_question, 
                resp_path= resp_path, 
                shadow_segments=target_shadow_segments)
            
            args.logger.info(f"Data Idx {' '.join([str(i) for i in args.target_data_indices])}: PREFIX: {repr(entry['prefix'])[1:-1]}, POSTFIX: {repr(entry['postfix'])[1:-1]}, LOSS:{entry['loss']:.4f}, TEST_ASV: {test_asv:.4f}")
        
            with open(csv_path, "a") as eval_f:
                writer = csv.writer(eval_f)
                writer.writerow(
                    [entry['seed'],attack, " ".join([str(i) for i in args.target_data_indices]), entry['prefix'], entry['postfix'],entry['loss'], test_asv] )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gcg hyparameter tester", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target_dataset",
                        type=str,
                        choices=TARGET_DATASETS,
                        default='amazon_reviews',
                        help="Target dataset")
    parser.add_argument("--injected_dataset",
                        type=str,
                        default='print',
                        choices=INJECT_DATASETS,
                        help="Inject dataset")
    parser.add_argument("--optim",
                        type=str,
                        nargs='+',
                        choices=['obliinjection', 'combine_attack','nanogcg_w_new_loss','nanogcg_wo_new_loss','gcg_wo_new_loss','neuralexec_w_new_loss',
                        'neuralexec_wo_new_loss','judgedeceiver','obliinjection_ce'],
                        help="optimizer used")
    parser.add_argument('--seed',
                        type=int,
                        help='manual seed to set')
    parser.add_argument('--target_data_indices', 
                        nargs='+',
                        type=int, 
                        default=[50])
    parser.add_argument('--injected_data_indices', 
                        type=int, 
                        nargs='+', 
                        default=[10])
    parser.add_argument('--attack_permute_num',
                        type=int,
                        default=1)
    parser.add_argument('--model',
                        type=str,
                        choices=['llama3','llama-3.1-8b-instruct', 'mistral-7b-instruct-v0.3','qwen-2.5-7b-instruct-1m', "falcon3-7b-instruct", 'qwen3-4b-instruct', 'llama4'],
                        default='llama3')
    parser.add_argument("--iter_num",
                        type=int,
                        default=500)
    parser.add_argument("--selected_num",
                        type=int,
                        default=100,
                        help="number of reviews")
    parser.add_argument("--eval_permute_num",
                        type=int)
    parser.add_argument('--gpus', 
                        nargs='+', 
                        type=int, 
                        help='index of visible gpus')
    parser.add_argument("--log_path",
                        default="log",
                        type=str)
    parser.add_argument("--review_lst_num",   
                        type=int,
                        default=10)
    parser.add_argument("--str2print",
                        type=str)
    parser.add_argument("--keyword",
                        type=str)
    parser.add_argument("--json_file",
                        type=str)
    parser.add_argument("--use_shadow_segment",
                        type=bool,
                        default=False,)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    
    seed_everything(args.seed)

    args.target_data_num = len(args.target_data_indices)
    args.injected_data_num = len(args.injected_data_indices)

    assert (args.str2print and args.keyword) or (not args.str2print and not args.keyword), "str2print and keyword should be used together"
    main(args)