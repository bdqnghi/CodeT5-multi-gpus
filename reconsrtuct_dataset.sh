#! /bin/bash

LANGUAGE="javascript"
OUTPUT_DIR=codexglue_sep27

python convert_data_format.py \
    --language ${language} \
    --filepath /media/Z/data_new/hungtq29/codet5/data/summarize/javascript/train.jsonl \
    --prompt_language \
    --target_dir ${output_dir} \
    --target_filename train \
    --min_length 3

python convert_data_format.py \
    --language ${language} \
    --filepath /media/Z/data_new/hungtq29/codet5/data/summarize/javascript/valid.jsonl \
    --prompt_language \
    --target_dir ${output_dir} \
    --target_filename  valid \
    --min_length 3

python convert_data_format.py \
    --language ${language} \
    --filepath /media/Z/data_new/hungtq29/codet5/data/summarize/javascript/test.jsonl \
    --prompt_language \
    --target_dir ${output_dir} \
    --target_filename test \
    --min_length 3
