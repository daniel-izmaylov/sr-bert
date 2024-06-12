from solvers import DialogBERTSolver
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from tqdm import tqdm
import numpy as np
import torch
import nltk
import gc

import models, solvers, data_loader
from argparse import Namespace

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)        
 

class DialogBert_Infernce():
    def  __init__(self,lexicon=0, with_feature=0,IMSR=1,number_labels=3):

        args ={'data_path': './data/',"lexicon":0,"IMSR":IMSR,"number_labels":number_labels, 'dataset': 'Sahar_labeld', 'fine_tune': 1,"train_base":0, 'model': 'DialogBERT', 'model_size': 'base', 'language': 'hebrew', 'per_gpu_train_batch_size': 16, 'per_gpu_eval_batch_size': 32, 'grad_accum_steps': 1, 'learning_rate': 5e-05, 'weight_decay': 0.01, 'adam_epsilon': 1e-08, 'max_grad_norm': 1.0, 'n_epochs': 20.0, 'max_steps': -1, 'warmup_steps': 1, 'version': 'unbalanced_7_long', 'version_load': 'pretrain/noFreezing_30K_long', 'reload_from': 67500, 'max_num_utts': 40, 'early_stop': -1.0, 'logging_steps': 9999, 'validating_steps': 265, 'save_steps': 1500, 'save_total_limit': 2, 'seed': 42, 'fp16': False, 'fp16_opt_level': 'O1', 'local_rank': -1, 'server_ip': '', 'server_port': '',"feature_loss_alpha":1.0, "with_turns":1}
        args["lexicon"]=lexicon
        args["with_feature"]=with_feature
        args= Namespace(**args)
        args.data_path = os.path.join(args.data_path, args.dataset)


        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.n_gpu = torch.cuda.device_count()
            # print(f"number of gpus: {args.n_gpu}")
            # if args.n_gpu>0:
        args.device = device
        set_seed(args)
        self.args=args
        self.solver= solvers.DialogBERTSolver(self.args)

    def return_model(self,model_version="Unbalanced_3",reload_from=1500,model_size="base"):
        if reload_from==None:
            #check what is the last model
            output_dir = os.path.join(f"/home/izmaylov/Thesis/SR_BERT/output/{self.args.model}/{model_size}/classification/{model_version}/models/")
            # all the folders in the output dir
            all_folders = glob.glob(output_dir + "/*")
            best_temp=0
            for folder in all_folders:
                # get the number of the model
                temp = int(folder.split("--")[-1])
                if temp > best_temp:
                    best_temp = temp
            best_temp= reload_from
            print("reload from: ",best_temp) 
        return self.solver.load_fine_tune(self.args,model_version=model_version,reload_from=reload_from,model_size=model_size)



# t=DialogBert_Infernce(lexicon=1, with_feature=1,with_turns=1 ,IMSR=0 ,number_labels=2)
# model=t.return_model(model_version="SSK_4.5_longer",reload_from="BestModel--1400",model_size="base",IMSR=1 ,number_labels=3)
