# This script converts Dung's data format to
# the correct format for training and inference
# code summarization model, including summarizing
# at function-level and parameter-level

import sys
import os
import argparse
import json
from tqdm import tqdm
from utils import load_js, check_len, create_data_sample, write_samples

def query_param_docstring(param_docstring):
    if isinstance(param_docstring, dict):
        return param_docstring.get("docstring", None)
    elif isinstance(param_docstring, str):
        return param_docstring.strip()
    if param_docstring is None or param_docstring in ["", "None", "return None", "returns None"]:
         return None

def process_datapoint(args, line, idx, write_stream):
    count_none = 0
    summaries = []
    js = load_js(line, idx)
    if js is None:
        return 0
    docstring_params = js.pop("docstring_params", dict())

    js_main = create_data_sample(js, "function", args.language, prompt_language=args.prompt_language)
    if check_len(args, js_main["docstring_tokens"]):
        summaries.append(js_main)
    for param_name, param_docstring in docstring_params.items():
        if param_name in ["other_param"]:
            continue
        param_docstring = query_param_docstring(param_docstring)
        if param_docstring is None:
            count_none += 1
            continue

        param_docstring = param_docstring.split("\n", maxsplit=1)[0]
        if check_len(args, param_docstring):
            if param_name.lower() in ["return", "returns"]:
                param_name = param_name.replace("s","").strip()
                js_param = create_data_sample(js, 
                                              "return", 
                                              args.language, 
                                              docs=param_docstring, 
                                              prompt_language=args.prompt_language,
                                              prefix_target_sequence=args.prefix_target,
                                              )
            else:
                js_param = create_data_sample(js, 
                                              "param", 
                                              args.language, 
                                              docs=param_docstring, 
                                              param_name=param_name, 
                                              prompt_language=args.prompt_language,
                                              prefix_target_sequence=args.prefix_target,
                                              )
            summaries.append(js_param)
    write_samples(write_stream, summaries)
    return count_none

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, choices=["python", "java", "javascript", "ruby", "php", "go"], required=True)
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--prompt_language", action="store_true")
    parser.add_argument("--file_indexing", type=int, default=2)
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--target_filename", type=str, default=None)
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--prefix_target", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    filename, ext = os.path.splitext(args.filepath)
    source_dir, filename = os.path.split(filename)

    if args.target_dir is not None:
        os.makedirs(args.target_dir, exist_ok=True)
        target_dir = args.target_dir
    else:
        target_dir = source_dir

    target_filename = args.target_filename if args.target_filename is not None else "{}{}".format(filename, args.file_indexing)

    with open(args.filepath, encoding="utf-8") as f, open("{}/{}{}".format(target_dir, target_filename, ext), "w", encoding="utf-8") as f1:
        count_none = 0
        for idx, line in tqdm(enumerate(f), total=100000):
            count_none += process_datapoint(args, line, idx, f1)
        print("count_none", count_none)

if __name__ == "__main__":
    main()
