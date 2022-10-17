import sys
sys.path.append("../")
import argparse
import re
import json
from tqdm import tqdm
import numpy as np
from data.summarize.commentutils import tokenize_docstring
from transformers import RobertaTokenizer
from pprint import pprint

# SEP_TOKEN = "</s>"


def create_data_sample(js_object, data_type, language, docs=None, param_name=None, prompt_language=False):
    assert data_type in ["function", "param", "return"]
    js = js_object.copy()
    js["prompt_tokens"] = ["<{}>".format(data_type)]
    if prompt_language:
        js["prompt_tokens"] = ["<{}>".format(language)] + js["prompt_tokens"]
    if data_type == "function": 
        js["comment_type"] = "main_content"
    else:
        assert docs is not None
        #if language in ["ruby", "java", "javascript", "php"]:
        docs = prefix_target(data_type, docs, param_name, symbol=True)
        js["docstring_tokens"] = tokenize_docstring(docs)
        if data_type == "param":
            assert param_name is not None
            js["comment_type"] = "param"
            js["prompt_tokens"].append(param_name)
        elif data_type == "return":
            js["comment_type"] = "return"
    js["code_tokens"] = js["prompt_tokens"] + [SEP_TOKEN] + js["code_tokens"]
    return js

def write_samples(data_stream, summaries):
    return_idx = -1
    for sen_idx, summary in enumerate(summaries):
        if summary["comment_type"] == "return":
            return_idx = sen_idx
            break
    if return_idx > -1:
        return_obj = summaries.pop(return_idx)
        summaries.append(return_obj)
    for sen_idx, summary in enumerate(summaries):
        summary["sentence_id"] = sen_idx
        json.dump(summary, data_stream, indent=4)
        data_stream.write("\n")

def process_docsparam(args, summaries, docsparam, verbose=False):
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
        if len(param_content.strip().split()) > 2:
            js_param = create_data_sample(js, 
                                          "param", 
                                          args.language, 
                                          docs=param_content, 
                                          param_name=param_name, 
                                          prompt_language=args.prompt_language)
            summaries.append(js_param)

def process_docsreturn(args, summaries, docsreturn, verbose=False):
    docsreturn = docsreturn.strip().split("\n")[0]
    docsreturn = re.sub("\{.*\}", "", docsreturn)
    if len(docsreturn.strip().split()) > 2:
        js_return = create_data_sample(js, "return", args.language, docs=docsreturn, prompt_language=args.prompt_language)
        summaries.append(js_return)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--language", type=str, required=True, choices=["java", "javascript", "php"], nargs="+")
    parser.add_argument("--language", type=str, required=True, choices=["java", "javascript", "php", "python"])
    parser.add_argument("--prompt_language", action="store_true")
    parser.add_argument("--file_indexing", type=int, default=2)

    args = parser.parse_args()

    for language in args.languages:
        num_train_samples = {#"java": 373425, 
                             "java": 164923,
                             "javascript": 58025,
                             "php": 251820,
                             }
        splits = ["train", "valid", "test"]
        verbose = False
        for split in tqdm(splits):
            with open(f"data/summarize/{language}/{split}.jsonl", "r", encoding="utf-8") as f1, \
                 open(f"data/summarize/{language}/{split}{args.file_indexing}.jsonl", "w", encoding="utf-8") as f2:
                prompt_lengths = []
                for idx, line in tqdm(enumerate(f1), total=num_train_samples[language]):
                    if verbose:
                        if idx > 10:
                            break
                    summaries = []
                    sentence_id = 0
                    line = line.strip()
                    js = json.loads(line)
                    js["sample_id"] = idx
                    js_main = create_data_sample(js, "function", language, prompt_language=args.prompt_language)
                    summaries.append(js_main)

                    # process return & parameters
                    docstring = js["docstring"].strip()
                    docstring = docstring.replace("@returns", "@return").replace("@params", "@param")
                    if verbose:
                        print("="*20, idx, "="*20)
                        print(docstring)
                        print('-'*10)
                    docsparams = docstring.split("@return", maxsplit=1)
                    if len(docsparams) == 1:
                        docsparams = docsparams[0]
                    elif len(docsparams) == 2:
                        docsparams, docsreturn = docsparams
                        process_docsreturn(args, summaries, docsreturn, verbose=verbose)
                        # docsreturn = docsreturn.strip().split("\n")[0]
                        # docsreturn = re.sub("\{.*\}", "", docsreturn)
                        # if len(docsreturn.strip().split()) > 2:
                        #     js_return = create_data_sample(js, "return", language, docs=docsreturn, prompt_language=args.prompt_language)
                        #     summaries.append(js_return)
                    else:
                        pprint(docsparams)
                        raise Exception("Error")
                    docsparams = docsparams.split("@param")[1:]
                    for docsparam in docsparams:
                        process_docsparam(args, summaries, docsparam, verbose=verbose)
                        # docsparam = docsparam.strip().split("\n")[0]
                        # docsparam = re.sub("\{.*\}", "", docsparam)
                        # if docsparam.strip() != "":
                        #     split_param_content = docsparam.strip().split(maxsplit=1)
                        #     if language == "php":
                        #         if len (split_param_content) == 1:
                        #             continue
                        #         if len(split_param_content) == 2:
                        #             phrase1, phrase2 = split_param_content
                        #             if "$" in phrase1:
                        #                 param_name, param_content = phrase1, phrase2
                        #             else:
                        #                 split_param_content_ = phrase2.strip().split(maxsplit=1)
                        #                 if len(split_param_content_) == 1:
                        #                     continue
                        #                 if len(split_param_content_):
                        #                     param_type = phrase1
                        #                     param_name, param_content = split_param_content_
                        #     else:
                        #         if len(split_param_content) == 1:
                        #             param_name, param_content = split_param_content[0], ''
                        #         elif len(split_param_content) == 2:
                        #             param_name, param_content = split_param_content
                        #         else:
                        #             if verbose:
                        #                 print(split_param_content)
                        #             # break
                        #             continue
                        #     if len(param_content.strip().split()) > 2:
                        #         js_param = create_data_sample(js, "param", language, docs=param_content, param_name=param_name, prompt_language=args.prompt_language)
                        #         summaries.append(js_param)
                    write_samples(f2, summaries)
                if verbose:
                    print("min", min(prompt_lengths))
                    print("max", max(prompt_lengths))
                    print("mean", np.mean(prompt_lengths))
                    print("median", np.quantile(prompt_lengths, 0.5))
                    print("25%", np.quantile(prompt_lengths, 0.25))
                    print("75%", np.quantile(prompt_lengths, 0.75))
                    print("90%", np.quantile(prompt_lengths, 0.90))

if __name__ == "__main__":
    main()
