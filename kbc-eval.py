#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import torch
from functools import partial
from tqdm.auto import tqdm
import time,datetime
from datetime import timedelta

import pandas as pd  # working with dataset
import math, re  # working with regex
from collections import Counter  # for compute metrics

import numpy as np # for instance loss

import warnings
warnings.simplefilter('once', UserWarning) # Only show warning once
warnings.simplefilter('once', DeprecationWarning)

stop_by_first_period = False

# ============================== Metrics Functions ============================


def compute_score(predictions, labels):
    f1 = exact_match = total = 0
    for i in range(len(predictions)):
        total += 1
        prediction = predictions[i]
        label = labels[i]
        # print(prediction, " | ", label)
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, label)
        f1 += metric_max_over_ground_truths(f1_score, prediction, label)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return exact_match, f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    scores_for_ground_truths = []
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def exact_match_score(prediction, ground_truth):  # sentence level
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):  # token level
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        f1 = 0
        precision = -1
        recall = -1
    else:
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    # dealing with accented?
    def remove_accented(text):
        import unicodedata

        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    def expand_contractions(text):
        raise NotImplementedError

    # this one may not need
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
        return re.sub(pat, "", text)

    def lower(text):
        return text.lower()

    return lower(white_space_fix(remove_articles(remove_punc(s))))



# ============================== Metrics Functions ============================


def compute_score_new(predictions, labels):
    f1 = exact_match = total = 0
    for i in range(len(predictions)):
        total += 1
        prediction = predictions[i]
        label = labels[i]

        if stop_by_first_period:
        # for llama-2 querying with natural language text
        # stop by first period sign, this indicate the answer is complete
            prediction = prediction.split('.')[0]

        # deal with multiple answer candidate
        if len(label.split(","))==1:
            exact_match += metric_max_over_ground_truths(exact_match_score, prediction, label)
            f1 += metric_max_over_ground_truths(f1_score, prediction, label)
        else:
            possible_answers = label.split(",")
            # print(prediction, " | ", possible_answers)
            ems, f1s = [], []
            for ans in possible_answers:
                ems.append(metric_max_over_ground_truths(exact_match_score, prediction, ans))
                f1s.append(metric_max_over_ground_truths(f1_score, prediction, ans))
            if 1 in ems:
                idx = ems.index(1)
            else:
                idx = f1s.index(max(f1s))
            exact_match += ems[idx]
            f1 += f1s[idx]

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return exact_match, f1

# ======================= Main Function ================


def parse_args():
    parser = argparse.ArgumentParser(description="compute em/f1 for missing fact zeroshot considering multiple answer candidate")
    
    parser.add_argument(
        "--file_name", type=str, default=None, help="A csv file of generated missing fact for kbc evaluation."
    )
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # if VERBOSE_DEBUG:
    for key, value in args.__dict__.items():
        print("'{}': {},".format(key, value))

    results_df = pd.read_csv(args.file_name)
    results_df.fillna('', inplace=True)
    targets_df = results_df["target"]
    targets = targets_df.tolist()
    predicts_df = results_df["prediction"]
    predicts = predicts_df.tolist()

    # targets = targets[:5]
    # predicts = predicts[:5]

    print("For File {}".format(args.file_name))

    og_exact_match, og_f1_score = compute_score(predicts, targets)
    print("\tthe original scores are em={}, f1={}".format(og_exact_match, og_f1_score))


    new_em_score, new_f1_score = compute_score_new(predicts, targets)
    print("\tthe recalculated scores are em={}, f1={}".format(new_em_score, new_f1_score))
    

if __name__ == "__main__":
    main()