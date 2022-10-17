import re
from typing import List, Dict, Any, Set, Optional
import json

DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")

def tokenize_docstring(docstring: str) -> List[str]:
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstring) if t is not None and len(t) > 0]

def load_js(line, idx):
    line = line.strip()
    try:
        js = json.loads(line)
    except json.decoder.JSONDecodeError:
        return None
    js["sample_id"] = idx
    if "processed_docstring" in js:
        js["docstring"] = js.pop("processed_docstring")
    if "processed_docstring_tokens" in js:
        js["docstring_tokens"] = js.pop("processed_docstring_tokens")
    return js

def check_len(args, docstring):
    if isinstance(docstring, str):
        return len(docstring.strip().split()) >= args.min_length
    elif isinstance(docstring, list):
        return len(docstring) >= args.min_length

def prefix_target(data_type, docs, param_name=None, symbol=True):
    if data_type == "return":
        docs = "{} {}".format(data_type, docs)
    elif data_type == "param":
        assert param_name is not None
        docs = "{} {} {}".format(data_type, param_name, docs)
    if symbol:
        docs = "@{}".format(docs)
    return docs

def create_data_sample(js_object, 
                       data_type, 
                       language, 
                       docs=None, 
                       param_name=None, 
                       prompt_language=False,
                       prefix_target_sequence=False):
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
        if prefix_target_sequence:
            docs = prefix_target(data_type, docs, param_name, symbol=True)
        js["docstring_tokens"] = tokenize_docstring(docs)
        if data_type == "param":
            assert param_name is not None
            js["comment_type"] = "param"
            js["prompt_tokens"].append(param_name)
        elif data_type == "return":
            js["comment_type"] = "return"
    # js["code_tokens"] = js["prompt_tokens"] + [SEP_TOKEN] + js["code_tokens"]
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
        json.dump(summary, data_stream)
        data_stream.write("\n")
