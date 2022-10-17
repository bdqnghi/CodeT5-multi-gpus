#! /bin/bash

OUTPUT_DIR=output_dirs/train
CACHE_DIR=${OUTPUT_DIR}/cache__data
RES_DIR=${OUTPUT_DIR}

mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 29502 test_run_ddp.py \
--do_train \
--do_eval \
--do_eval_bleu \
--do_test \
--task summarize \
--sub_task multi \
--model_type codet5 \
--train_filename 'example_javascript_dataset.jsonl' \
--dev_filename 'example_javascript_dataset.jsonl' \
--test_filename 'example_javascript_dataset.jsonl' \
--tokenizer_name 'Salesforce/codet5-small' \
--model_name_or_path 'Salesforce/codet5-small' \
--data_dir ./data \
--cache_path ${CACHE_DIR}  \
--output_dir ${OUTPUT_DIR}  \
--summary_dir tensorboard \
--res_dir ${RES_DIR} \
--res_fn ${RES_DIR}/results.txt \
--train_batch_size 96 \
--eval_batch_size 96 \
--max_source_length 256 \
--max_target_length 140 \
--num_train_epochs 1 \
--learning_rate 5e-5 \
--patience 5 \
--warmup 1000 \
--gradient_accumulation_steps 1 \
--target_update_manner none \
--test_beam_sizes 10 \
--max_prompt_length 15 \
--min_target_length 5 \
--distributed
