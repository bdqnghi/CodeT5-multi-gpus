# language data -> tensor file -> partition tensor file
import sys
sys.path.append("../")
import os
import json
import argparse
import multiprocessing
import torch
from torch.utils.data import TensorDataset

from utils_temp import load_and_cache_gen_data
from models import MODEL_CLASSES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--tokenizer_name", default="codet5-small", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument('--target_update_manner', type=str, choices=['none', 'first_epoch', 'epoch', 'step'])
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_prompt_length", default=15, type=int)
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--filepaths", type=str, nargs="+", required=True)
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--num_partitions", type=int, default=3)

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.cache_path, exist_ok=True)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    tokenizer = MODEL_CLASSES[args.model_type][-1].from_pretrained(args.tokenizer_name)
    for filepath in args.filepaths:
        split_tag = "train"
        examples, data = load_and_cache_gen_data(args, 
                                                filepath, 
                                                pool, 
                                                tokenizer, 
                                                split_tag, 
                                                only_src=False)
        with open(os.path.join(args.cache_path, "num_samples.txt"), "w") as f:
            f.write(str(len(data)))

        source_ids, target_ids = data.tensors
        assert source_ids.size(0) == target_ids.size(0)
        num_samples_per_partition = len(data) // args.num_partitions
        for idx in range(args.num_partitions):
            cache_fn = os.path.join(args.cache_path, "partition{}.pt".format(idx))
            start_idx = idx * num_samples_per_partition
            end_idx = (idx + 1) * num_samples_per_partition if idx + 1 < args.num_partitions else len(data)
            source_ids_ = source_ids[start_idx:end_idx]
            target_ids_ = target_ids[start_idx:end_idx]
            data_ = TensorDataset(source_ids_, target_ids_) 
            torch.save(data_, cache_fn)

if __name__ == "__main__":
    main()
