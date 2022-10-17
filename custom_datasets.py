import os
import random 
import platform
import json
from pathlib import Path
import logging
import itertools
from tree_sitter import Parser, Language
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
from transformers import RobertaTokenizer
from evaluator import smooth_bleu

from pprint import pprint

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
#fileHandler = logging.FileHandler("./valid.{}".format(__file__).replace("py", "log"))
#logger.addHandler(fileHandler)

def get_language(language):
    home = str(Path.home())
    cd = os.getcwd()
    plat = platform.system()
    p = os.path.join(home, ".tree-sitter", "bin")
    file = f'{language}.so'
    return Language(os.path.join(p, file), language)

language = get_language('ruby')
parser = Parser()
parser.set_language(language)

def match_from_span(node, blob: str) -> str:
    lines = blob.split('\n')
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
    else:
        return lines[line_start][char_start:char_end]

def extract_variables(node, results):
    if node.type == "method":
        for n in node.children:
            if n.type == "identifier":
                results['method'] = n
                break
    elif node.type == "method_parameters":
        for n in node.children:
            if n.type == "identifier":
                results['variables'].append(n)
    elif node.type == 'assignment':
        for n in node.children:
            if n.type == "identifier":
                results['variables'].append(n)
                break
    for n in node.children:
        extract_variables(n, results)

def get_variable_names(code_snippet):
    tree = parser.parse(code_snippet.encode())
    results = {"variables": []}
    extract_variables(tree.root_node, results)
    if "method" in results.keys():
        results["method"] = match_from_span(results["method"], code_snippet)
    if "variables" in results.keys():
        results["variables"] = [match_from_span(node, code_snippet) for node in results["variables"]]
    return results

def augment_code_snippet1(example, replace_function_name=False, replacement_rate=0.15):
    code_snippet, code_tokens, variable_names = example["original_string"], example["code_tokens"], example["variable_names"]
    token_mapping = dict()
    var_index = 0
    for k, v in variable_names.items():
        if replace_function_name and k == "method":
            if random.uniform(0,1) < replacement_rate:
                token_mapping[v] = "function_name"
        elif k == "variables":
            for var_name in set(v):
                if random.uniform(0,1) < replacement_rate:
                    var_index += 1
                    token_mapping[var_name] = "var{}".format(var_index)

    for i, token in enumerate(code_tokens):
        code_tokens[i] = token_mapping.get(token, token)

    return code_tokens

def augment_code_snippet2(example, all_variable_names, replace_function_name=False, replacement_rate=0.15):
    if replacement_rate == 0.:
        return example["code_tokens"]
    code_snippet, code_tokens, variable_names = example["original_string"], example["code_tokens"], example["variable_names"]
    token_mapping = dict()
    sampled_tokens = []
    for k, v in variable_names.items():
        if replace_function_name and k == "method":
            if random.uniform(0,1) < replacement_rate:
                sampled_token = random.sample(all_variable_names, 1)[0]
                all_variable_names.remove(sampled_token)
                token_mapping[v] = sampled_token
                sampled_tokens.append(sampled_token)
        elif k == "variables":
            for var_name in set(v):
                if random.uniform(0,1) < replacement_rate:
                    sampled_token = random.sample(all_variable_names, 1)[0]
                    all_variable_names.remove(sampled_token)
                    token_mapping[var_name] = sampled_token
                    sampled_tokens.append(sampled_token)

    all_variable_names = all_variable_names.union(sampled_tokens)
    #print(token_mapping)

    for i, token in enumerate(code_tokens):
        code_tokens[i] = token_mapping.get(token, token)

    return code_tokens
    
class SummarizeDataset(Dataset):
    def __init__(self, args, filename, tokenizer, split_tag):
        assert split_tag in ["train", "test"]
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.split_tag = split_tag
        self.data = []
        self.all_variable_names = set()
        with open(filename, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                js = json.loads(line)
                if "idx" not in js:
                    js["idx"] = idx
                if args.replacement_rate > 0:
                    variable_names = get_variable_names(js["original_string"])
                    js["variable_names"] = variable_names
                    if "method" in variable_names.keys():
                        self.all_variable_names.add(variable_names["method"])
                    if "variables" in variable_names.keys():
                        self.all_variable_names = self.all_variable_names.union(variable_names["variables"])
                self.data.append(js)

        #self.data = self.data[:100]
        self.num_labeled_data = len(self.data)
        
        if args.unlabeled_data_filename is not None:
            with open(args.unlabeled_data_filename, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    js = json.loads(line)
                    if "idx" not in js:
                        js["idx"] = idx
                    if args.replacement_rate > 0:
                        variable_names = get_variable_names(js["original_string"])
                        if "method" in variable_names.keys():
                            self.all_variable_names.add(variable_names["method"])
                        if "variables" in variable_names.keys():
                            self.all_variable_names = self.all_variable_names.union(variable_names["variables"])
                    #js["docstring_tokens"] = "unlabeled data"
                    self.data.append(js)

        #self.data = self.data[:200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        if self.split_tag == "train":
            #code_snippet = example["original_string"]
            #code_tokens = augment_code_snippet1(example, replace_function_name=self.args.replace_function_name, replacement_rate=self.args.replacement_rate)
            #print(example["code_tokens"])
            code_tokens = augment_code_snippet2(example, self.all_variable_names, replace_function_name=self.args.replace_function_name, replacement_rate=self.args.replacement_rate)
            #print(example["code_tokens"])
            docstring_tokens = example["docstring_tokens"]
            target_str = " ".join(docstring_tokens).replace("\n", " ")
            target_str = " ".join(target_str.strip().split())
        else:
            code_tokens = example["code_tokens"]

        source_str = " ".join(code_tokens).replace("\n", " ")
        source_str = " ".join(source_str.strip().split())

        if self.args.sub_task != "none":
            source_str = "{} {}: {}".format(self.args.task, self.args.sub_task, source_str)
        else:
            source_str = "{}: {}".format(self.args.task, source_str)
        source_str = source_str.replace("</s>", "<unk>")
        source_ids = self.tokenizer.encode(source_str, max_length=self.args.max_source_length, padding="max_length", truncation=True)
        assert source_ids.count(self.tokenizer.eos_token_id) == 1
        if self.split_tag == "test":
            return torch.LongTensor(source_ids)
        else:
            if self.args.add_lang_ids:
                target_str = "<en>" + target_str
            target_str = target_str.replace("</s>", "<unk>")
            target_ids = self.tokenizer.encode(target_str, max_length=self.args.max_target_length, padding="max_length", truncation=True)
            assert target_ids.count(self.tokenizer.eos_token_id) == 1
            return torch.LongTensor(source_ids), torch.LongTensor(target_ids)

class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (primary_batch + secondary_batch
                for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                       grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

def get_pseudo_args():
    import argparse
    from configs import add_args
    from models import build_or_load_gen_model
    parser = argparse.ArgumentParser()
    return add_args(parser)

def test_SummarizeDataset():
    args = get_pseudo_cli_args()
    config, model, tokenizer = build_or_load_gen_model(args)
    model.eval()
    train_dataset = SummarizeDataset(args, filename=args.train_filename, tokenizer=tokenizer, split_tag="train")
    for i in range(5):
        train_dataset.args.replacement_rate = 0.
        source_ids, target_ids = train_dataset[i]
        target_str = tokenizer.decode(target_ids, skip_special_tokens=True)
        logger.info("target: {}".format(target_str))
        preds = model.generate(source_ids.unsqueeze(0), 
                max_length=128, 
                return_dict_in_generate=True, 
                output_scores=True, 
                #do_sample=True, top_p=0.05
                )
        generated_ids = preds.sequences.squeeze(0)
        #seq_score = -sum([torch.log_softmax(t.squeeze(), dim=-1)[idx] for idx, t in zip(generated_ids, preds.scores)]) / torch.sum(preds.sequences != 0).item()
        seq_score = -sum([torch.log_softmax(t.squeeze(), dim=-1).sum().item() for idx, t in zip(generated_ids, preds.scores)]) / torch.sum(preds.sequences != 0).item()
        prediction_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
        bleu_score = smooth_bleu.bleu([target_str], prediction_str)[0]
        logger.info("replacement_rate = 0: {}".format(prediction_str))
        logger.info("bleu: {}".format(bleu_score))
        logger.info("log probability: {}".format(seq_score))

        train_dataset.args.replacement_rate = 0.5
        source_ids, target_ids = train_dataset[i]
        preds = model.generate(source_ids.unsqueeze(0), max_length=128, return_dict_in_generate=True, output_scores=True, 
                #do_sample=True, top_p=0.05
                )
        generated_ids = preds.sequences.squeeze(0)
        #seq_score = -sum([torch.log_softmax(t.squeeze(), dim=-1)[idx] for idx, t in zip(generated_ids, preds.scores)]) / torch.sum(preds.sequences != 0).item()
        seq_score = -sum([torch.log_softmax(t.squeeze(), dim=-1).sum().item() for idx, t in zip(generated_ids, preds.scores)]) / torch.sum(preds.sequences != 0).item()
        prediction_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
        bleu_score = smooth_bleu.bleu([target_str], prediction_str)[0]
        logger.info("replacement_rate = 0.5: {}".format(prediction_str))
        logger.info("bleu: {}".format(bleu_score))
        logger.info("log probability: {}".format(seq_score))

        train_dataset.args.replacement_rate = 1.
        source_ids, target_ids = train_dataset[i]
        preds = model.generate(source_ids.unsqueeze(0), max_length=128, return_dict_in_generate=True, output_scores=True, 
                #do_sample=True, top_p=0.05
                )
        generated_ids = preds.sequences.squeeze(0)
        #seq_score = -sum([torch.log_softmax(t.squeeze(), dim=-1)[idx] for idx, t in zip(generated_ids, preds.scores)]) / torch.sum(preds.sequences != 0).item()
        seq_score = -sum([torch.log_softmax(t.squeeze(), dim=-1).sum().item() for idx, t in zip(generated_ids, preds.scores)]) / torch.sum(preds.sequences != 0).item()
        prediction_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
        bleu_score = smooth_bleu.bleu([target_str], prediction_str)[0]
        logger.info("replacement_rate = 1.: {}".format(prediction_str))
        logger.info("bleu: {}".format(bleu_score))
        logger.info("log probability: {}".format(seq_score))
        logger.info("*"*10)

def test_TwoStreamBatchSampler():
    args = get_pseudo_args()
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    dataset = SummarizeDataset(args, args.train_filename, tokenizer, "train")
    #primary_indices = range(dataset.num_labeled_data)
    #secondary_indices = range(dataset.num_labeled_data, len(dataset))
    primary_indices = list(range(dataset.num_labeled_data))
    secondary_indices = list(range(dataset.num_labeled_data, len(dataset)))
    batch_sampler = TwoStreamBatchSampler(primary_indices, secondary_indices, args.train_batch_size, args.unlabeled_data_batch_size)
    train_dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    for example in train_dataloader:
        print(example)
        print(len(example))
        print(type(example))
        print(example[0].shape)
        print(example[1].shape)
        for target in example[1][-args.unlabeled_data_batch_size:]:
            print(tokenizer.decode(target, skip_special_tokens=True))
        break

if __name__ == "__main__":
    #test_SummarizeDataset()
    test_TwoStreamBatchSampler()
