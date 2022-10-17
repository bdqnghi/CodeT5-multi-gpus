import json

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1

    # SETTING 3 modified from here
    assert args.max_prompt_length <= args.max_target_length - 2 # <bos>, <eos>, prompt tokens are included in target tokens
    prompt_tokens = tokenizer.tokenize(example.prompt)[:args.max_prompt_length]
    prompt_padding_length = args.max_prompt_length - len(prompt_tokens)
    prompt_tokens = prompt_tokens + [tokenizer.pad_token]*prompt_padding_length
    if stage == 'test':
        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        target_ids = prompt_ids
        #target_ids = prompt_ids + [tokenizer.bos_token_id]
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_tokens = tokenizer.tokenize(target_str)[:args.max_target_length - args.max_prompt_length - 2]
        target_tokens = prompt_tokens + [tokenizer.bos_token] + target_tokens + [tokenizer.eos_token]
        target_padding_length = args.max_target_length - len(target_tokens)
        target_tokens += [tokenizer.pad_token]*target_padding_length
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids=target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids=None,
                 hypos_ids=None,
                 risks=None,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.hypos_ids = hypos_ids
        self.risks = risks
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 prompt=None,
                 sample_id=None,
                 sentence_id=None,
                 comment_type=None,
                 url=None,
                 task='',
                 sub_task='',
                 hypos=None,
                 risks=None,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.prompt = prompt
        self.sample_id = sample_id
        self.sentence_id = sentence_id
        self.comment_type = comment_type
        self.url = url
        self.task = task
        self.sub_task = sub_task
        self.hypos = hypos
        self.risks = risks


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


#def read_summarize_examples(filename, data_num):
#    """Read examples from filename."""
#    examples = []
#    with open(filename, encoding="utf-8") as f:
#        for idx, line in enumerate(f):
#            line = line.strip()
#            js = json.loads(line)
#            if 'idx' not in js:
#                js['idx'] = idx
#            code = ' '.join(js['code_tokens']).replace('\n', ' ')
#            code = ' '.join(code.strip().split())
#            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
#            nl = ' '.join(nl.strip().split())
#            examples.append(
#                Example(
#                    idx=idx,
#                    source=code,
#                    target=nl,
#                )
#            )
#            if idx + 1 == data_num:
#                break
#    return examples

def read_summarize_examples(filename, data_num, split_tag, args=None):
    """Read examples from filename."""
    if isinstance(filename, str):
        filename = [filename]
    data_num_each_file = data_num // len(filename) if data_num != -1 else -1
    examples = []
    global_idx = 0
    for fname in filename:
        with open(fname, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # if split_tag == "train" and idx == 500:
                #     break
                # elif idx == 100:
                # # if idx == 100:
                #     break
                line = line.strip()
                js = json.loads(line)
                if args is not None and len(js["docstring_tokens"]) < args.min_target_length:
                    continue
                js['idx'] = global_idx
                code = ' '.join(js['code_tokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())
                nl = ' '.join(js['docstring_tokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
                prompt = " ".join(js["prompt_tokens"]).replace("\n", "")
                prompt = " ".join(prompt.strip().split())
                examples.append(
                    Example(
                        idx=global_idx,
                        source=code,
                        target=nl,
                        prompt=prompt,
                        sample_id=js["sample_id"],
                        sentence_id=js["sentence_id"],
                        comment_type=js["comment_type"]
                    )
                )
                global_idx += 1
                if idx + 1 == data_num_each_file:
                    break
    return examples

def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
