# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import math
import random
import re
import numpy as np
from tqdm import tqdm
from glob import glob
import multiprocessing
import time
import json
import gc
from collections import defaultdict

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data, load_dataset_from_examples
from ddp_utils import is_main_process, init_distributed, get_tqdm
from configs import add_args, set_seed, set_dist
from custom_datasets import SummarizeDataset, TwoStreamBatchSampler
from _utils import Example

if is_main_process():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)


def get_optimizer(args, model, num_train_data):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_train_optimization_steps = args.num_train_epochs * num_train_data // args.train_batch_size
    if args.scheduler_last_epoch != -1:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps,
                                                last_epoch=args.scheduler_last_epoch)
    return optimizer, scheduler

def train(args, batch, tokenizer, model):
    source_ids, target_ids = batch
    source_ids = source_ids.to(args.device)
    batch_size = source_ids.shape[0]

    target_ids = target_ids.to(args.device)
    source_mask = source_ids.ne(tokenizer.pad_token_id)

    if 1:
        # this worked, but train loss and ppl are high
        # bleu for param is around 38
        decoder_input_ids = torch.zeros_like(target_ids, dtype=target_ids.dtype, device=target_ids.device)
        decoder_input_ids[..., 1:] = target_ids[..., :-1].clone()
        decoder_input_ids[..., 0] = tokenizer.pad_token_id

        target_mask = target_ids.ne(tokenizer.pad_token_id)

        target_ids[..., :args.max_prompt_length] = -100
        target_ids.masked_fill_(target_ids == tokenizer.pad_token_id, -100)
    if 0:
        # this worked on bleu score, but train loss and valid ppl is much higher than other settings
        # bleu for snippet and return is high, but param is only around 32 to 33 # this is for old version
        # ****new version: this worked for main_content and param, but return is 2 scores lower than the above settting (30 compared to 32)
        decoder_input_ids = torch.zeros_like(target_ids, dtype=target_ids.dtype, device=target_ids.device)
        decoder_input_ids[..., 1:] = target_ids[..., :-1].clone()
        decoder_input_ids[..., 0] = tokenizer.pad_token_id

        target_mask = torch.ones_like(target_ids, dtype=source_mask.dtype, device=target_ids.device)
        target_mask[..., 1:args.max_prompt_length + 1] = target_ids[..., :args.max_prompt_length].ne(tokenizer.pad_token_id)
        target_mask[..., args.max_prompt_length:] *= target_ids[..., args.max_prompt_length:].ne(tokenizer.pad_token_id)
        # target_ids = target_ids[..., 1:].contiguous()
        # target_mask = target_ids.ne(tokenizer.pad_token_id)

        target_ids[..., :args.max_prompt_length] = -100
        target_ids.masked_fill_(target_ids == tokenizer.pad_token_id, -100)

    if 0:
        decoder_input_ids = target_ids.clone()
        decoder_input_ids = decoder_input_ids[..., :-1]
        target_ids = target_ids[...,1:].contiguous()
        target_mask = target_ids.ne(tokenizer.pad_token_id)
        target_ids[..., :args.max_prompt_length - 1] = -100
        target_ids.masked_fill_(target_ids == tokenizer.pad_token_id, -100)
        if 0:
            target_ids_[..., :args.max_prompt_length] = -100
            target_ids = target_ids_[..., 1:].contiguous()
        
    outputs = model(input_ids=source_ids, 
                    attention_mask=source_mask,
                    labels=target_ids, 
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=target_mask)
    return outputs.loss

def train_mrt(args, batch, tokenizer, model):
    assert args.num_sequences_to_sample <= args.kd_extraction_beam_size
    if args.num_sequences_to_sample == 0:
        args.num_sequences_to_sample = args.kd_extraction_beam_size
    source_ids, indices = batch
    source_ids = source_ids.to(args.device)
    batch_size = source_ids.shape[0]
    hypos_ids, risks = sample_sequences(args, 
                                        source_ids, 
                                        indices,
                                        train_examples, 
                                        model, 
                                        tokenizer)
    model.train()
    if args.num_sequences_to_sample < args.kd_extraction_beam_size:
        risks, sorted_indices = risks.topk(k=args.num_sequences_to_sample, 
                                           dim=-1) 
        hypos_ids = hypos_ids.gather(1, sorted_indices.unsqueeze(2).repeat(1, 1, args.max_target_length))
    hypos_ids = hypos_ids.view(hypos_ids.size(0), -1).to(args.device)
    risks = risks.to(args.device) if args.num_sequences_to_sample > 1 else None # sequence-interpolation in knowledge distillation
    source_mask, hypos_mask = source_ids.ne(tokenizer.pad_token_id), hypos_ids.ne(tokenizer.pad_token_id)

    assert args.model_type != "roberta"
    outputs = model(input_ids=source_ids, 
                    attention_mask=source_mask,
                    labels=hypos_ids,
                    decoder_attention_mask=hypos_mask,
                    risks=risks)
    return outputs.loss


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = DistributedSampler(eval_data, shuffle=False) if args.distributed else SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, 
                                 sampler=eval_sampler, 
                                 batch_size=args.eval_batch_size,
                                 num_workers=4, 
                                 pin_memory=True)
    if is_main_process():
        # Start evaluating model
        logger.info("  " + "***** Running ppl evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    #for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
    for batch in get_tqdm(eval_dataloader, desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch

        source_mask = source_ids.ne(tokenizer.pad_token_id)

        decoder_input_ids = torch.zeros_like(target_ids, dtype=target_ids.dtype, device=target_ids.device)
        decoder_input_ids[..., 1:] = target_ids[..., :-1].clone()
        decoder_input_ids[..., 0] = tokenizer.pad_token_id

        # target_mask = torch.zeros_like(target_ids, dtype=source_mask.dtype, device=target_ids.device)
        # target_mask[..., 1:args.max_prompt_length + 1] = target_ids[..., :args.max_prompt_length].ne(tokenizer.pad_token_id)
        # target_mask[..., args.max_prompt_length:] += target_ids[..., args.max_prompt_length:].ne(tokenizer.pad_token_id)
        # target_ids = target_ids[..., 1:].contiguous()
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        target_ids[..., :args.max_prompt_length] = -100
        target_ids.masked_fill_(target_ids == tokenizer.pad_token_id, -100)

        with torch.no_grad():
            outputs = model(input_ids=source_ids, 
                            attention_mask=source_mask,
                            labels=target_ids, 
                            decoder_attention_mask=target_mask)
            loss = outputs.loss
            if args.distributed:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
                loss /= args.n_gpus

    if is_main_process():
        eval_loss += loss.item()
        batch_num += 1
        eval_loss = eval_loss / batch_num
        eval_ppl = round(np.exp(eval_loss), 5)
        return eval_ppl

def extract_knowledge_step(args, source_ids, indices, examples, model, tokenizer, last_max_score_strs=None, last_max_scores=None):
    model.eval()
    source_ids = source_ids.to(args.device)
    source_mask = source_ids.ne(tokenizer.pad_token_id)
    target_strs = [examples[k].target for k in indices.cpu().numpy()]
    pred_strs, pred_scores = [], []
    # only support sequence-interpolation mode
    with torch.no_grad():
        preds = model.generate(source_ids,
                               attention_mask=source_mask,
                               use_cache=True,
                               num_beams=args.kd_extraction_beam_size,
                               early_stopping=args.task == 'summarize',
                               max_length=args.max_target_length,
                               num_return_sequences=args.kd_extraction_beam_size
                               )
        pred_strs = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in preds.cpu().numpy()]
        for h, target in enumerate(target_strs):
            sample_scores = []
            if args.extraction_metric == "bleu":
                for k in range(args.kd_extraction_beam_size):
                    pred = pred_strs[h*args.kd_extraction_beam_size+k]
                    bleu_score = smooth_bleu.bleu([target], pred)[0]
                    sample_scores.append(bleu_score)
            elif args.extraction_metric == "codebleu":
                beam_preds = pred_strs[h*args.kd_extraction_beam_size:(h+1)*args.kd_extraction_beam_size]
                sample_scores = calc_code_bleu.get_codebleu_for_kd_extraction(target, beam_preds, args.lang)
                sample_scores = [1 if beam_pred.strip().split() == target.strip().split() else sample_score for (sample_score, beam_pred) in zip(sample_scores, beam_preds)]
            pred_scores.append(sample_scores)
    max_scores = [max(sample_scores) for sample_scores in pred_scores]
    max_score_indices = [np.argmax(sample_scores) for sample_scores in pred_scores]
    max_score_strs = [pred_strs[i*args.kd_extraction_beam_size + idx] for i, idx in enumerate(max_score_indices)]
    total_em = 0
    for i, (max_score, max_score_str) in enumerate(zip(max_scores, max_score_strs)):
        target = target_strs[i]
        total_em += int(target.strip().split() == max_score_str.strip().split())
    num_old_strs = 0
    if last_max_scores is not None:
        max_scores = [max(max_score, last_max_score) for max_score, last_max_score in zip(max_scores, last_max_scores)]
        max_score_strs = [max_score_str if max_score > last_max_score else last_max_score_str for \
                          max_score_str, last_max_score_str, max_score, last_max_score in \
                          zip(max_score_strs, last_max_score_strs, max_scores, last_max_scores)]
        num_old_strs = sum([last_max_score > max_score for max_score, last_max_score in zip(max_scores, last_max_scores)])
    return max_score_strs, max_scores, num_old_strs, total_em


def extract_knowledge_epoch(args, examples, source_dataloader, model, tokenizer, last_max_score_strs=None, last_max_scores=None):
    batch_size = source_dataloader.batch_size
    dataset_strs, dataset_scores = [], []
    em = 0
    num_old_strs = 0
    bar = tqdm(source_dataloader, total=len(source_dataloader), desc="Predicting labels")
    for j, batch in enumerate(bar):
        source_ids, indices = batch
        batch_last_max_score_strs, batch_last_max_scores = None, None
        if last_max_scores is not None:
            batch_last_max_score_strs = last_max_score_strs[j*batch_size:(j+1)*batch_size]
            batch_last_max_scores = last_max_scores[j*batch_size:(j+1)*batch_size]
        batch_strs, batch_scores, batch_num_old_strs, batch_em = extract_knowledge_step(\
                                                                 args, source_ids, indices, examples, model, tokenizer, \
                                                                 last_max_score_strs=batch_last_max_score_strs, \
                                                                 last_max_scores=batch_last_max_scores)
        dataset_strs.extend(batch_strs)
        dataset_scores.extend(batch_scores)
        num_old_strs += batch_num_old_strs
        em += batch_em
        bar.set_description("Score {} EM {}".format(round(np.mean(dataset_scores), 4), round(em/len(dataset_scores), 4)))
    prediction_ids = [tokenizer.encode(pred_str, max_length=args.max_target_length, padding="max_length", truncation=True) for pred_str in dataset_strs]
    prediction_ids = torch.tensor(prediction_ids, dtype=torch.long)
    source_ids = source_dataloader.dataset.tensors[0]
    train_data = TensorDataset(source_ids, prediction_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    metrics_value = np.mean(dataset_scores)
    if is_main_process():
        logger.info("{} = {}".format(args.extraction_metric, metrics_value))
        logger.info("Old predictions to be kept = {}".format(num_old_strs/len(dataset_strs)))

    if args.task in ["summarize"]:
        objs = [{'code_tokens': ex.source.split(), 'docstring_tokens': nl.split()} for ex, nl in zip(examples, dataset_strs)]
    elif args.task in ["concode"]:
        objs = [{'nl': ex.source.strip(), 'code': code.strip()} for ex, code in zip(examples, dataset_strs)]

    # save predictions to disk
    with open(f"{args.output_dir}/dataset_scores.json", "w") as f:
        json.dump(dataset_scores, f)
    if args.task in ["translate"]:
        src_objs = [ex.source.strip() for ex in examples]
        trg_objs = [code.strip() for code in dataset_strs]
        with open(f"{args.output_dir}/{args.sub_task.split('-')[0]}.{args.task}_{args.sub_task}_pseudo_labels.jsonl", "w") as f_src, open(f"{args.res_dir}/{args.sub_task.split('-')[1]}.{args.task}_{args.sub_task}_pseudo_labels.jsonl", "w") as f_trg:
            for src_obj, trg_obj in zip(src_objs, trg_objs):
                f_src.write(src_obj)
                f_src.write("\n")
                f_trg.write(trg_obj)
                f_trg.write("\n")
    else:
        with open(f"{args.output_dir}/{args.task}_pseudo_labels.jsonl", "w") as f:
            for obj in objs:
                json.dump(obj, f)
                f.write('\n')
    return train_dataloader, dataset_strs, dataset_scores

def sample_sequences(args, source_ids, indices, examples, model, tokenizer):
    model.eval()
    source_mask = source_ids.ne(tokenizer.pad_token_id)
    if args.subset_sampling_method == "beam_search":
        preds = model.generate(source_ids,
                               attention_mask=source_mask,
                               use_cache=True,
                               num_beams=args.kd_extraction_beam_size,
                               early_stopping=args.task == "summarize",
                               max_length=args.max_target_length,
                               num_return_sequences=args.kd_extraction_beam_size
                               )
    elif args.subset_sampling_method == "random":
        preds = model.generate(source_ids,
                               attention_mask=source_mask,
                               use_cache=True,
                               num_beams=1,
                               top_k=args.top_k,
                               do_sample=True,
                               temperature=args.temperature,
                               early_stopping=args.task == "summarize",
                               max_length=args.max_target_length,
                               num_return_sequences=args.kd_extraction_beam_size
                               )
    else:
        raise NotImplementedError

    pred_strs = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    hypos_ids = [tokenizer.encode(pred_str, max_length=args.max_target_length, padding="max_length", truncation=True) for pred_str in pred_strs]
    hypos_ids = torch.tensor(hypos_ids, dtype=torch.long).view(source_ids.size(0), args.kd_extraction_beam_size, -1)

    target_strs = [examples[i].target for i in indices.cpu().numpy()]
    
    risks = []
    for h, target in enumerate(target_strs):
        sample_risks = []
        if args.extraction_metric == "bleu":
            for k in range(args.kd_extraction_beam_size):
                pred = pred_strs[h*args.kd_extraction_beam_size+k]
                bleu_score = smooth_bleu.bleu([target], pred)[0]
                sample_risks.append(bleu_score)
        elif args.extraction_metric == "codebleu":
            beam_preds = pred_strs[h*args.kd_extraction_beam_size:(h+1)*args.kd_extraction_beam_size]
            sample_risks = calc_code_bleu.get_codebleu_for_kd_extraction(target, beam_preds, args.lang)
        risks.append(sample_risks)
    risks = torch.tensor(risks, dtype=torch.float)
    
    if args.add_target_to_hypos:
        target_ids = [tokenizer.encode(target_str, max_length=args.max_target_length, padding="max_length", truncation=True) for target_str in target_strs]
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        hypos_ids = torch.cat([target_ids.unsqueeze(1), hypos_ids], dim=1)
        risks = torch.cat([torch.ones(risks.size(0), 1, dtype=torch.float), risks], dim=1)

    return hypos_ids, risks


def eval_sequence_probability(args, eval_data, eval_examples, model, tokenizer, split_tag):
    if is_main_process():
        logger.info("  ***** Running sequence probability evaluation on {} data*****".format(split_tag))
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    dataset_sequence_probs = []
    with torch.no_grad():
        for j, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Eval sequence probability for {} set".format(split_tag)):
            source_ids = batch[0].to(args.device)
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            outputs = model.generate(source_ids,
                                     attention_mask=source_mask,
                                     use_cache=True,
                                     num_beams=1,
                                     num_return_sequences=1,
                                     early_stopping=args.task == "summarize",
                                     max_length=args.max_target_length,
                                     return_dict_in_generate=True,
                                     output_scores=True)
            scores = torch.cat([score.unsqueeze(1) for score in outputs.scores], dim=1) # [bz, seq_len, vocab_size]
            log_probs = torch.log_softmax(scores, dim=-1)
            log_probs = torch.gather(log_probs, dim=2, index=outputs.sequences[:,1:].unsqueeze(2)).squeeze()
            log_probs = log_probs * outputs.sequences[:,1:].ne(tokenizer.pad_token_id).float() \
                                  * outputs.sequences[:,1:].ne(tokenizer.eos_token_id).float()
            sequence_probs = log_probs.sum(-1).exp()
            dataset_sequence_probs.extend(list(sequence_probs.cpu().numpy()))
    return dataset_sequence_probs


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria, beam_size):
    if is_main_process():
        logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

    batch_size = args.eval_batch_size
    if beam_size == 1:
        batch_size *= 2
    eval_sampler = get_sampler(args, eval_data, split_tag="test")
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, 
                                     sampler=eval_sampler, 
                                     batch_size=batch_size,
                                     num_workers=4, 
                                     pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, 
                                     sampler=eval_sampler, 
                                     batch_size=batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    # for j, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
    for j, batch in enumerate(get_tqdm(eval_dataloader, desc="Eval bleu for {} set".format(split_tag))):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        decoder_input_ids = batch[1].to(args.device)
        decoder_input_ids = F.pad(decoder_input_ids, (1, 0), mode="constant", value=tokenizer.pad_token_id)
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=source_mask.dtype, device=decoder_input_ids.device)
        decoder_attention_mask[..., :-1] = decoder_input_ids[..., 1:].ne(tokenizer.pad_token_id) # this is for target_mask = target_ids.ne(tokenizer.pad_token_id)
        # decoder_attention_mask[..., 1:] = decoder_input_ids[..., 1:].ne(tokenizer.pad_token_id) # this is for masking in the correct position, not shifted
        kwargs = {"decoder_input_ids": decoder_input_ids,
                  "decoder_attention_mask": decoder_attention_mask}

        inference_module = model.module if args.distributed else model
        with torch.no_grad():
            preds = inference_module.generate(source_ids,
                                              attention_mask=source_mask,
                                              use_cache=True,
                                              num_beams=beam_size,
                                              early_stopping=args.task == 'summarize',
                                              max_length=args.max_target_length,
                                              **kwargs,
                                              )

            if args.distributed:
                max_length = torch.LongTensor([preds.size(1)]).cuda()
                dist.all_reduce(max_length, op=dist.ReduceOp.MAX)
                max_length = max_length.item()
                preds = F.pad(preds, (0, max_length - preds.size(1)), mode="constant", value=tokenizer.pad_token_id)

                if is_main_process():
                    tensor_list = [torch.zeros_like(preds, dtype=preds.dtype, device=preds.device) for _ in range(args.world_size)]
                    dist.gather(preds, tensor_list, dst=0)
                    dim = tensor_list[0].size(1)
                    preds = torch.stack(tensor_list, dim=1)
                    preds = preds.view(-1, preds.size(-1))
                else:
                    dist.gather(preds, dst=0)

            if is_main_process(): 
                top_preds = list(preds.cpu().numpy())
                pred_ids.extend(top_preds)

    if is_main_process():
        pred_ids = pred_ids[:len(eval_data)]
        pred_nls = [tokenizer.decode(id[args.max_prompt_length + 1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
        prompts_ = [tokenizer.decode(id[1:args.max_prompt_length + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
        # pred_nls = [tokenizer.decode(id, skip_special_tokens=False, clean_up_tokenization_spaces=False) for id in pred_ids]
        print(pred_nls[:100])
        pred_nls_by_type = {"main_content": defaultdict(list), "param": defaultdict(list), "return": defaultdict(list)}

        for prompt_, pred_nl, truth_ex in zip(prompts_, pred_nls, eval_examples):
            pred_nls_by_type[truth_ex.comment_type]["truth"].append(truth_ex)
            pred_nls_by_type[truth_ex.comment_type]["pred"].append(pred_nl)
            pred_nls_by_type[truth_ex.comment_type]["prompt"].append(prompt_)

        result = {}

        for comment_type, pred_dict in pred_nls_by_type.items():
            output_fn = os.path.join(args.res_dir, "test_{}.output".format(comment_type))
            gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(comment_type))
            src_fn = os.path.join(args.res_dir, "test_{}.src".format(comment_type))
            dev_accs, predictions = [], []
            with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
                for sample_id, (prompt_, pred_nl, gold) in enumerate(zip(pred_dict["prompt"], pred_dict["pred"], pred_dict["truth"])):
                    if comment_type == "param":
                        assert gold.prompt[:len(prompt_)] == prompt_, "prompt: {}, prompt_: {}".format(gold.prompt, prompt_)
                    else:
                        assert gold.prompt == prompt_, "prompt: {}, prompt_: {}".format(gold.prompt, prompt_)
                    dev_accs.append(pred_nl.strip() == gold.target.strip())
                    # for smooth-bleu4 evaluation
                    predictions.append(str(sample_id) + '\t' + pred_nl)
                    f.write(str(sample_id) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(sample_id) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(sample_id) + '\t' + gold.source.strip() + '\n')

            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            try:
                bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            except ZeroDivisionError:
                bleu = -1
            result["em_{}".format(comment_type)] = np.mean(dev_accs) * 100
            result["bleu_{}".format(comment_type)] = bleu
            result["count_{}".format(comment_type)] = len(predictions)

        return result


def get_sampler(args, data, split_tag):
    if split_tag in ["train"] and args.target_update_manner in ["none", "step"]:
        if args.distributed:
            return DistributedSampler(data,
                                         shuffle=True,
                                         seed=args.seed,
                                         drop_last=True,
                                         )
        else:
            return RandomSampler(data) 
    elif split_tag in ["dev", "test"] or args.target_update_manner in ["first_epoch", "epoch"]:
        if args.distributed:
            return DistributedSampler(data,
                                         shuffle=False,
                                         seed=args.seed,
                                         drop_last=False,
                                        )
        else:
            return SequentialSampler(data)


def prepare_training_data_loader(args, tokenizer, pool, examples=None):
    train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, "train")
    train_sampler = get_sampler(args, train_data, split_tag="train")
    batch_size = args.train_batch_size if args.target_update_manner in ["none", "step"] else args.eval_batch_size
    train_dataloader = DataLoader(train_data, 
                                  sampler=train_sampler, 
                                  batch_size=batch_size,
                                  num_workers=4, 
                                  pin_memory=True)
    return train_examples, train_data, train_dataloader

def train_epoch(args, 
                model, 
                train_dataloader, 
                optimizer, 
                scheduler, 
                epoch, 
                tokenizer, 
                # global_step, 
                # glob_steps,
                **kwargs,
                ):
    global_step, glob_steps = kwargs["global_step"], kwargs["glob_steps"]
    if args.distributed:
        train_dataloader.sampler.set_epoch(epoch)

    bar = get_tqdm(train_dataloader, desc="Training")
    nb_tr_examples, nb_tr_steps, tr_loss, tr_cons_loss, tr_cl_loss = 0, 0, 0, 0, 0
    model.train()
    for step, batch in enumerate(bar):
        if args.target_update_manner in ["none", "first_epoch", "epoch"]:
            loss = train(args, batch, tokenizer, model)
        elif args.target_update_manner in ["step"]:
            loss = train_mrt(args, batch, tokenizer, model)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        nb_tr_examples += batch[0].size(0)
        nb_tr_steps += 1
        glob_steps += 1
        loss.backward()
        if args.distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= args.world_size

        if nb_tr_steps % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if is_main_process():
                global_step += 1
                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4) 
                bar.set_description("[{}] Train loss {}".format(epoch, round(train_loss, 3)))
        #dist.barrier()

    kwargs["global_step"], kwargs["glob_steps"] = global_step, glob_steps
    return kwargs

# if args.do_eval:

def eval_epoch(
        args,
        model,
        epoch,
        dev_dataset,
        tokenizer,
        pool,
        flag_tensor,
        tb_writer=None,
        fa=None,
        **kwargs,
    ):
    global_step, glob_steps, best_bleu_em, best_ppl, not_bleu_em_inc_cnt, not_loss_dec_cnt = \
            kwargs["global_step"], kwargs["glob_steps"], kwargs["best_bleu_em"], kwargs["best_ppl"], kwargs["not_bleu_em_inc_cnt"], kwargs["not_loss_dec_cnt"]
    # Eval model with dev dataset
    if 'dev_loss' in dev_dataset:
        eval_examples, eval_data = dev_dataset['dev_loss']
    else:
        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
        dev_dataset['dev_loss'] = eval_examples, eval_data

    eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
    if args.distributed:
        dist.barrier()
    if is_main_process():
        result = {'epoch': epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            fa.write("  %s = %s\n" % (key, str(result[key])))
        logger.info("  " + "*" * 20)
        if args.data_num == -1:
            tb_writer.add_scalar('dev_ppl', eval_ppl, epoch)

        # save last checkpoint
        if args.save_last_checkpoints:
            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Save the last model into %s", output_model_file)

        if eval_ppl < best_ppl:
            not_loss_dec_cnt = 0
            logger.info("  Best ppl:%s", eval_ppl)
            logger.info("  " + "*" * 20)
            fa.write("[%d] Best ppl changed into %.4f\n" % (epoch, eval_ppl))
            best_ppl = eval_ppl

            # Save best checkpoint for best ppl
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if args.always_save_model:
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("Save the best ppl model into %s", output_model_file)
        else:
            not_loss_dec_cnt += 1
            logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
            if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                    epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                logger.info(early_stop_str)
                fa.write(early_stop_str)
                if args.distributed:
                    flag_tensor += 1
                else:
                    kwargs["early_stopping"] = True
                    return kwargs
                    # return True, None
        not_loss_dec_cnt = args.patience + 1 # do not care about loss
        logger.info("***** CUDA.empty_cache() *****")

    # early stopping for data distributed parallel training
    if args.distributed:
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
        if flag_tensor == 1:
            kwargs["early_stopping"] = True
            return kwargs
        dist.barrier()

    torch.cuda.empty_cache()
    gc.collect()
    if args.do_eval_bleu:
        eval_examples, eval_data = load_and_cache_gen_data(args,
                                                           args.dev_filename, 
                                                           pool, 
                                                           tokenizer, 
                                                           #'dev',
                                                           "test",
                                                           only_src=False, 
                                                           is_sample=True)

        if is_main_process():
            logger.info("***** CUDA.empty_cache() *****")
            logger.info('***** Evaluating model *****')
        result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % epoch, args.validation_beam_size)
        dist.barrier()
        if is_main_process():
            dev_bleu, dev_em = 0, 0
            for comment_type in ["main_content", "param", "return"]:
                bleu_score_ = result["bleu_{}".format(comment_type)]
                if bleu_score_ >= 0:
                    dev_bleu += result["bleu_{}".format(comment_type)] * result["count_{}".format(comment_type)]
                    dev_em += result["em_{}".format(comment_type)] * result["count_{}".format(comment_type)]
            dev_bleu, dev_em = dev_bleu / len(eval_data), dev_em / len(eval_data)
            logger.info("***** CUDA.empty_cache() *****")
            logger.info("***** Eval results *****")
            for comment_type in ["main_content", "param", "return"]:
                logger.info("  dev_%s_bleu = %s", comment_type, str(round(result["bleu_{}".format(comment_type)], 4)))
            logger.info("  dev_bleu = %s", str(round(dev_bleu, 4)))
            logger.info("  dev_em = %s", str(round(dev_em, 4)))
            if args.task in ['summarize']:
                dev_bleu_em = dev_bleu
            elif args.task in ['defect']:
                dev_bleu_em = dev_em
            else:
                dev_bleu_em = dev_bleu + dev_em
            if args.data_num == -1:
                tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, epoch)
            if dev_bleu_em > best_bleu_em:
                not_bleu_em_inc_cnt = 0
                logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                epoch, dev_bleu_em, dev_bleu, dev_em)
                logger.info("  " + "*" * 20)
                best_bleu_em = dev_bleu_em
                fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                epoch, best_bleu_em, dev_bleu, dev_em))
                # Save best checkpoint for best bleu
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if args.data_num == -1 or args.always_save_model:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best bleu model into %s", output_model_file)
            else:
                not_bleu_em_inc_cnt += 1
                logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                fa.write(
                "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                    epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                    stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                        epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                    logger.info(stop_early_str)
                    fa.write(stop_early_str)
                    if args.distributed:
                        flag_tensor += 1
                    else:
                        kwargs["early_stopping"] = True
                        return kwargs

        # early stopping for data distributed parallel training
        if args.distributed:
            dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
            if flag_tensor:
                kwargs["early_stopping"] = True
                return kwargs
            dist.barrier()

    kwargs["best_bleu_em"], kwargs["best_ppl"], kwargs["not_bleu_em_inc_cnt"], kwargs["not_loss_dec_cnt"], kwargs["early_stopping"] = best_bleu_em, best_ppl, not_bleu_em_inc_cnt, not_loss_dec_cnt, False
    return kwargs

def get_tensor_filepaths(args):
    data_files_by_language = []
    for data_dir in args.data_dirs:
        # files_and_dirs = os.listdir(data_dir)
        files_and_dirs = glob("{}/*".format(data_dir))
        filepaths = [filepath for filepath in files_and_dirs if os.path.isfile(filepath) and re.search("partition[0-9].pt", filepath)]
        random.shuffle(filepaths)
        data_files_by_language.append(filepaths)

    num_langauges = len(data_files_by_language)
    num_partitions_per_language = len(data_files_by_language[0])
    assert all([len(filepaths) == num_partitions_per_language for filepaths in data_files_by_language])
    
    for filepaths in zip(*data_files_by_language):
        yield filepaths

def prepare_training_data_loader_by_partition(filepaths):
    source_ids = []
    target_ids = []
    for filepath in filepaths:
        data = torch.load(filepath)
        # source_ids.append(data.tensors[0][:500])
        # target_ids.append(data.tensors[1][:500])
        source_ids.append(data.tensors[0])
        target_ids.append(data.tensors[1])
    source_ids = torch.concat(source_ids, dim=0)
    target_ids = torch.concat(target_ids, dim=0)

    train_data =  TensorDataset(source_ids, target_ids)
    
    train_sampler = get_sampler(args, train_data, split_tag="train")
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4,
                                  pin_memory=True
                                  )
    return train_dataloader, train_data

def get_num_train_data(args, train_data):
    if train_data is not None:
        return len(train_data)
    else:
        num_train_data = 0
        for data_dir in args.data_dirs:
            with open("{}/num_samples.txt".format(data_dir), "r") as f:
                num_train_data += int(f.read())
        return num_train_data

def main(args):
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    if args.distributed:
        model.cuda()
        model = DDP(model, 
                    device_ids=[args.local_rank])
    else:
        model.to(args.device)
    pool = multiprocessing.Pool(args.cpu_cont)
    train_filename, dev_filename, test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    if is_main_process():
        fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')
    else:
        fa = None
    test_filename = [test_filename]
    if args.train_filename is None:
        args.train_filename = train_filename
    if args.dev_filename is None:
        args.dev_filename = dev_filename
    if args.test_filename is None:
        args.test_filename = test_filename

    if args.do_train:
        flag_tensor = torch.zeros(1).cuda()
        #if args.local_rank in [-1, 0] and args.data_num == -1:
        if is_main_process():
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)
        else:
            tb_writer = None

        if args.data_dirs == "":
        # if args.num_partitions == 1:
            train_examples, train_data, train_dataloader = prepare_training_data_loader(args, tokenizer, pool)
            train_example_num = len(train_data)
        else:
            train_data, train_dataloader = None, None

        if args.target_update_manner in ["first_epoch", "epoch"]:
            source_dataloader = train_dataloader

        train_example_num = get_num_train_data(args, train_data)

        optimizer, scheduler = get_optimizer(args, model, train_example_num)

        # Start training
        if is_main_process():
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_example_num)
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
            logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        glob_steps = 0
        last_max_score_strs, last_max_scores = None, None
        kwargs = {"global_step": global_step, 
                  "glob_steps": glob_steps, 
                  "best_bleu_em": best_bleu_em, 
                  "best_ppl": best_ppl, 
                  "not_bleu_em_inc_cnt": not_bleu_em_inc_cnt,
                  "not_loss_dec_cnt": not_loss_dec_cnt,
                  "early_stopping": False,
                  }
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            if args.data_dirs == "":
                kwargs = train_epoch(args, 
                                     model, 
                                     train_dataloader, 
                                     optimizer, 
                                     scheduler, 
                                     cur_epoch, 
                                     tokenizer, 
                                     **kwargs,
                                    )
            else:
                for tensor_filepaths in get_tensor_filepaths(args):
                    train_dataloader, train_data = prepare_training_data_loader_by_partition(tensor_filepaths)
                    kwargs = train_epoch(args, 
                                         model, 
                                         train_dataloader, 
                                         optimizer, 
                                         scheduler, 
                                         cur_epoch, 
                                         tokenizer, 
                                         **kwargs,
                                        )
                    del train_dataloader
                    del train_data
                    gc.collect()

            if args.do_eval:
                kwargs = eval_epoch(args, 
                                    model, 
                                    cur_epoch, 
                                    dev_dataset, 
                                    tokenizer, 
                                    pool, 
                                    flag_tensor,
                                    tb_writer=tb_writer,
                                    fa=fa,
                                    **kwargs,
                                    )

            if kwargs["early_stopping"]: 
                break

            if is_main_process():
                logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
            gc.collect()

        #if args.local_rank in [-1, 0] and args.data_num == -1:
        if is_main_process():
            tb_writer.close()
            logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        if is_main_process():
            logger.info("  " + "***** Testing *****")
            logger.info("  Batch size = %d", args.eval_batch_size)

        for test_filename in args.test_filename:
            for criteria in ['best-bleu']:
                file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
                if is_main_process():
                    logger.info("Reload model from {}".format(file))
                    logger.info("Test filename {}".format(test_filename))
                    fa.write("Test filename {}\n".format(test_filename))
                if args.do_train:
                    map_location = {"cuda:0": "cuda:%d" % args.rank} if args.distributed else None
                    model.module.load_state_dict(torch.load(file, map_location=map_location))
                eval_examples, eval_data = load_and_cache_gen_data(args, 
                                                                   test_filename, 
                                                                   pool, 
                                                                   tokenizer, 
                                                                   'test',
                                                                   only_src=False, 
                                                                   is_sample=False)
                for beam_size in args.test_beam_sizes:
                    result = eval_bleu_epoch(args, 
                                             eval_data, 
                                             eval_examples, 
                                             model, 
                                             tokenizer, 
                                             'test',
                                             criteria, 
                                             beam_size)
                    dist.barrier()
                    if is_main_process():
                        result_str = "[beam_size: %d]\n" % (beam_size)
                        for k, v in result.items():
                            result_str += "\t{}: {}\n".format(k, v)
                        logger.info(result_str)
                        fa.write(result_str)
            if args.res_fn and is_main_process():
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)
            if args.eval_sequence_probability:
                # not support ddp yet
                sequence_probs = eval_sequence_probability(args, eval_data, eval_examples, model, tokenizer, 'test')
                if is_main_process():
                    logger.info("[Sequence probability: Greedy search] {}\n".format(np.mean(sequence_probs)))
                    fa.write("[Sequence probability: Greedy search] {}\n".format(np.mean(sequence_probs)))

        if is_main_process():
            logger.info("Finish and take {}".format(get_elapse_time(t0)))
            fa.write("Finish and take {}".format(get_elapse_time(t0)))
            fa.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    assert len(os.listdir(args.output_dir)) <= 1, "Output directory {} exists".format(args.output_dir)
    if is_main_process():
        logger.info(args)
    main(args)
