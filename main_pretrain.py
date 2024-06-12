import argparse
import glob
import logging
import os
import pickle
import random

from tqdm import tqdm
import numpy as np
import torch
import nltk
import gc



logger = logging.getLogger(__name__)
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)        
 

def main():
    # nltk.download('wordnet')

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path", default='./data/', type=str, help="The input data path.")
    parser.add_argument("--dataset", default='Sahar_30K_4.5', type=str, help="dataset name")
    parser.add_argument("--fine_tune", default=0, type=int, help="Fine tuning the model for classification  task")
    parser.add_argument("--train_base", default=0, type=int, help="Train the base model or not")
    parser.add_argument("--lexicon", default=0, type=int, help="Train the base model or not")
    parser.add_argument("--with_feature", default=7, type=int, help="Train the base model or not")
    parser.add_argument("--with_turns", default=1, type=int, help="Train the base model or not")
    parser.add_argument("--deepSSK", default=1, type=int, help="Train the base model or not")
    parser.add_argument("--model", default="DialogBERT", type=str, help="The model architecture to be fine-tuned.") 
    parser.add_argument("--model_size", default="base", type=str, help="tiny, small, base, large")
    parser.add_argument("--language", default="hebrew", type=str, help= "language, english or chinese")

    parser.add_argument("--reload_from", default=-1, type=int, help="The global iteration of optimal checkpoint.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--grad_accum_steps', type=int, default=2,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--n_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override n_epochs.")
    # parser.add_argument("--warmup_steps", default=10000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_steps", default=15000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--version", default=6, type=str, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_num_utts", default="150", type=int, help= "max number of utterances")
    parser.add_argument("--feature_loss_alpha", default="1", type=float, help= "max number of utterances")


    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--validating_steps', type=int, default = 78750, help= "Validate every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=78750, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=2,
                    help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    
    args = parser.parse_args()

    args.data_path = os.path.join(args.data_path, args.dataset)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print("number of gpus: ", args.n_gpu)
        # if args.n_gpu>0:
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    solver = getattr(solvers, args.model+'Solver')(args)

    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if not args.fine_tune:
        global_step, tr_loss = solver.train(args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
    elif args.fine_tune:
        global_step, tr_loss = solver.fine_tune(args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        

if __name__ == "__main__":
    print("Training starts")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    gc.collect()
    torch.cuda.empty_cache()


    main()
