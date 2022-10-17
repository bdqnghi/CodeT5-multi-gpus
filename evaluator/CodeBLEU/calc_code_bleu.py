# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
# https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU

# -*- coding:utf-8 -*-
import argparse
import os
from evaluator.CodeBLEU import bleu, weighted_ngram_match, syntax_match, dataflow_match

def get_codebleu_for_kd_extraction(ref, hyps, lang, params='0.25,0.25,0.25,0.25'):
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]
    reference = ref.strip()
    hypothesis = [x.strip() for x in hyps]
    
    tokenized_ref = reference.split()
    tokenized_hyps = [x.split() for x in hypothesis]

    ngram_match_scores = [bleu.sentence_bleu([tokenized_ref], tokenized_hyp) for tokenized_hyp in tokenized_hyps]

    root_dir = os.path.dirname(__file__)
    keywords = [x.strip() for x in open(root_dir + '/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_ref_with_weights = [tokenized_ref, make_weights(tokenized_ref, keywords)]

    weighted_ngram_match_scores = [weighted_ngram_match.sentence_bleu([tokenized_ref_with_weights], tokenized_hyp) for tokenized_hyp in tokenized_hyps]

    # calculate syntax match
    syntax_match_scores = syntax_match.corpus_syntax_match_kd_extraction(reference, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_scores = dataflow_match.corpus_dataflow_match_kd_extraction(reference, hypothesis, lang)

    code_bleu_scores = [alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score
                        for ngram_match_score, 
                            weighted_ngram_match_score, 
                            syntax_match_score, 
                            dataflow_match_score in zip(ngram_match_scores, 
                                                        weighted_ngram_match_scores, 
                                                        syntax_match_scores, 
                                                        dataflow_match_scores)
                      ]

    return code_bleu_scores

def get_codebleu(refs, hyp, lang, params='0.25,0.25,0.25,0.25'):
    if not isinstance(refs, list):
        refs = [refs]
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]

    # preprocess inputs
    pre_references = [[x.strip() for x in open(file, 'r', encoding='utf-8').readlines()] for file in refs]
    hypothesis = [x.strip() for x in open(hyp, 'r', encoding='utf-8').readlines()]

    for i in range(len(pre_references)):
        print(len(hypothesis))
        print(len(pre_references[i]))
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    root_dir = os.path.dirname(__file__)
    keywords = [x.strip() for x in open(root_dir + '/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'. \
          format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    return code_bleu_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', type=str, nargs='+', required=True,
                        help='reference files')
    parser.add_argument('--hyp', type=str, required=True,
                        help='hypothesis file')
    parser.add_argument('--lang', type=str, required=True,
                        choices=['java', 'js', 'c_sharp', 'php', 'go', 'python', 'ruby'],
                        help='programming language')
    parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
                        help='alpha, beta and gamma')

    args = parser.parse_args()
    code_bleu_score = get_codebleu(args.refs, args.hyp, args.lang, args.params)
    print('CodeBLEU score: ', code_bleu_score)

