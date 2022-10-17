# This script convert CodeXGLUE data format into
# valid format for running code summarization 
# including documenting both function-level and
# paramemter-level by prompting on target sequences

import sys
sys.path.append("../")
import os
import re
import json
import argparse
from tqdm import tqdm
from pprint import pprint
import numpy as np
from transformers import RobertaTokenizer
from utils import tokenize_docstring, load_js, check_len, create_data_sample, write_samples

def process_docsparam(args, js, summaries, docsparam, verbose=False):
    docsparam = docsparam.strip().split("\n")[0]
    docsparam = re.sub("\{.*\}", "", docsparam)
    if docsparam.strip() != "":
        split_param_content = docsparam.strip().split(maxsplit=1)
        if args.language == "php":
            if len (split_param_content) == 1:
                return
            if len(split_param_content) == 2:
                phrase1, phrase2 = split_param_content
                if "$" in phrase1:
                    param_name, param_content = phrase1, phrase2
                else:
                    split_param_content_ = phrase2.strip().split(maxsplit=1)
                    if len(split_param_content_) == 1:
                        return
                    if len(split_param_content_):
                        param_type = phrase1
                        param_name, param_content = split_param_content_
        else:
            if len(split_param_content) == 1:
                param_name, param_content = split_param_content[0], ''
            elif len(split_param_content) == 2:
                param_name, param_content = split_param_content
            else:
                if verbose:
                    print(split_param_content)
                return
        if check_len(args, param_content):
            js_param = create_data_sample(js, 
                                          "param", 
                                          args.language, 
                                          docs=param_content, 
                                          param_name=param_name, 
                                          prompt_language=args.prompt_language
                                          )
            summaries.append(js_param)

def process_docsreturn(args, js, summaries, docsreturn, verbose=False):
    docsreturn = docsreturn.strip().split("\n")[0]
    docsreturn = re.sub("\{.*\}", "", docsreturn)
    if check_len(args, docsreturn):
        js_return = create_data_sample(js, 
                                       "return", 
                                       args.language, 
                                       docs=docsreturn, 
                                       prompt_language=args.prompt_language
                                       )
        summaries.append(js_return)

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--languages", type=str, required=True, choices=["java", "javascript", "php"], nargs="+")
    parser.add_argument("--language", type=str, required=True, choices=["java", "javascript", "php", "python"])
    # parser.add_argument("--filepaths", type=str, nargs="+", required=True)
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--prompt_language", action="store_true")
    parser.add_argument("--file_indexing", type=int, default=3)
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--target_filename", type=str, default=None)
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")

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
    #for split in tqdm(splits):
    with open(args.filepath, "r", encoding="utf-8") as f, open("{}/{}{}".format(target_dir, target_filename, ext), "w", encoding="utf-8") as f1:
    # with open(f"{split}.jsonl", "r", encoding="utf-8") as f1, \
    #      open(f"{split}3.jsonl", "w", encoding="utf-8") as f2:
        prompt_lengths = []
        for idx, line in tqdm(enumerate(f), total=58025):
            if args.verbose:
                if idx > 10:
                    break
            summaries = []
            sentence_id = 0
            js = load_js(line, idx)
            # line = line.strip()
            # js = json.loads(line)
            # js["sample_id"] = idx
            js_main = create_data_sample(js, "function", args.language, prompt_language=args.prompt_language)
            # js["prompt_tokens"] = ["<function>"]
            # #js["code_tokens"] = prompt_tokens + js["code_tokens"]
            # js["comment_type"] = "main_content"
            # summaries.append(js)
            summaries.append(js_main)
            #js["code_tokens"] = js["code_tokens"][2:]
            docstring = js["docstring"].strip()
            docstring = docstring.replace("@returns", "@return").replace("@params", "@param")
            if args.verbose:
                print("="*20, idx, "="*20)
                print(docstring)
                print('-'*10)
            #if "@param" in docstring:
            docsparams = docstring.split("@return", maxsplit=1)
            if len(docsparams) == 1:
                docsparams = docsparams[0]
            #elif len(docsparams) > 1:
            #    pass
            elif len(docsparams) == 2:
                docsparams, docsreturn = docsparams
                process_docsreturn(args, js, summaries, docsreturn, verbose=args.verbose)
                # docsreturn = docsreturn.strip().split("\n")[0]
                # docsreturn = re.sub("\{.*\}", "", docsreturn)
                # if docsreturn.strip() != "":
                #     js_return = js.copy()
                #     docsreturn = "@return {}".format(docsreturn)
                #     js_return["docstring_tokens"] = tokenize_docstring(docsreturn)
                #     js_return["prompt_tokens"] = ["<return>"]
                #     js_return["comment_type"] = "return"
                #     summaries.append(js_return)
            else:
                pprint(docsparams)
                raise Exception("Error")
            docsparams = docsparams.split("@param")[1:]
            for docsparam in docsparams:
                process_docsparam(args, js, summaries, docsparam, verbose=args.verbose)
            #summaries = {" ".join(re.sub("\{.*\}", "", summary).strip().split()) for summary in summaries}
            write_samples(f1, summaries)
            # return_idx = -1
            # for sen_idx, summary in enumerate(summaries):
            #     if summary["comment_type"] == "return":
            #         return_idx = sen_idx
            #         break
            # if return_idx > -1:
            #     return_obj = summaries.pop(return_idx)
            #     summaries.append(return_obj)
            # for sen_idx, summary in enumerate(summaries):
            #     summary["sentence_id"] = sen_idx
            #     json.dump(summary, f2)
            #     f2.write("\n")
        if args.verbose:
            #for summary in summaries:
            #    #print(idx, " ".join(summary))
            #    print(idx, summary)
            print("min", min(prompt_lengths))
            print("max", max(prompt_lengths))
            print("mean", np.mean(prompt_lengths))
            print("median", np.quantile(prompt_lengths, 0.5))
            print("25%", np.quantile(prompt_lengths, 0.25))
            print("75%", np.quantile(prompt_lengths, 0.75))
            print("90%", np.quantile(prompt_lengths, 0.90))

if __name__ == "__main__":
    main()
