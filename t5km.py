#!/usr/bin/env python
# coding=utf-8

import argparse
import wandb
import os
import time
import datetime

import numpy as np
import torch
import torch.optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

import pandas as pd  # working with dataset
import math
import re  # working with regex
from collections import Counter  # for compute metrics

LARGE_DATASET_SIZE = 2**24
VERBOSE_NSAMPLE = 10
VERBOSE_MODE = False

LOGGING_STEP = 500
SAVING_EPOCH = 5

DO_INIT_EVAL = True

EARLY_STOPPING = True
EARLY_STOPPING_N_EPOCH = 10
EARLY_STOPPING_EM_SCORE = 99.5  # max 100
EARLY_STOPPING_F1_SCORE = 99.5

# ===================== Dataframe, Dataset, Dataloader ==========================


class KMemDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert knowledge triplet for text-to-text training")
            print(
                "\tsource - Subject: + subject_text + . Relation: + relation_text + . Object: ")
            print("\ttarget - object_text")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        subj_text = (self.dataframe).iloc[index, 0]
        rel_text = (self.dataframe).iloc[index, 1]
        obj_text = (self.dataframe).iloc[index, 2]
        source_text = "Subject: " + \
            str(subj_text) + ". Relation: " + str(rel_text) + ". Object: "
        target_text = str(obj_text)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class KMemNatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert knowledge triplet for text-to-text training")
            print("\tsource - subject_text + space  +  relation_text + . ")
            print("\ttarget - object_text")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        subj_text = (self.dataframe).iloc[index, 0]
        rel_text = (self.dataframe).iloc[index, 1]
        obj_text = (self.dataframe).iloc[index, 2]
        source_text = str(subj_text) + " " + str(rel_text) + "."
        target_text = str(obj_text)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class KMemFlexiSubjDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print(
                "convert knowledge triplet for text-to-text training - flexible to Subj prediction")
            print(
                "\tsource - Object: + object_text + . Relation: + relation_text + . Subject: ")
            print("\ttarget - subject_text")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        subj_text = (self.dataframe).iloc[index, 0]
        rel_text = (self.dataframe).iloc[index, 1]
        obj_text = (self.dataframe).iloc[index, 2]
        source_text = "Object: " + \
            str(obj_text) + ". Relation: " + str(rel_text) + ". Subject: "
        target_text = str(subj_text)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class KMemFlexiRelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print(
                "convert knowledge triplet for text-to-text training - flexible to relation prediction")
            print(
                "\tsource - Subject: + subject_text + . Object: + object_text + . Relation: ")
            print("\ttarget - relation_text")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        subj_text = (self.dataframe).iloc[index, 0]
        rel_text = (self.dataframe).iloc[index, 1]
        obj_text = (self.dataframe).iloc[index, 2]
        source_text = "Subject: " + \
            str(subj_text) + ". Object: " + str(obj_text) + ". Relation: "
        target_text = str(rel_text)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class OneHopForwardDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert onehop knowledge for forward reasoning EntA,Rel,EntB")
            print("\tsource - Subject: + EntA + . Relation: + Rel + . Object: ")
            print("\ttarget - EntB")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_ba = (self.dataframe).iloc[index, 3]
        source_text = "Subject: " + \
            str(ent_a) + ". Relation: " + str(rel_ab) + ". Object: "
        target_text = str(ent_b)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class OneHopBackwardDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert onehop knowledge for backward reasoning EntB,InvRel,EntA")
            print("\tsource - Subject: + EntB + . Relation: + InvRel + . Object: ")
            print("\ttarget - EntA")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_ba = (self.dataframe).iloc[index, 3]
        source_text = "Subject: " + \
            str(ent_b) + ". Relation: " + str(rel_ba) + ". Object: "
        target_text = str(ent_a)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class OneHopForwardNatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print(
                "convert onehop knowledge for forward reasoning EntA,Rel,EntB in natural language")
            print(
                "\tsource - natural language querying EntA + Rel with extra_id_0 marker")
            print("\ttarget - EntB")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_ba = (self.dataframe).iloc[index, 3]

        if rel_ab == "sibling":
            qus_text = "the sibling of {} is".format(ent_a)
        elif rel_ab == "shares border with":
            qus_text = "{} shares border with".format(ent_a)
        elif rel_ab == "father":
            qus_text = "the father of {} is".format(ent_a)
        elif rel_ab == "mother":
            qus_text = "the mother of {} is".format(ent_a)
        elif rel_ab == "capital":
            qus_text = "the capital of {} is".format(ent_a)
        elif rel_ab == "part of":
            qus_text = "{} is part of".format(ent_a)
        elif rel_ab == "country":
            qus_text = "the country {} belongs to is".format(ent_a)
        else:
            raise Exception("unexpected rel_ab={} from example A={},r={},B={},invr={}".format(
                rel_ab, ent_a, rel_ab, ent_b, rel_ba))
        ans_text = str(ent_b)

        source_text = "Fill in the blank: " + qus_text + " <extra_id_0>"
        target_text = ans_text

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class OneHopBackwardNatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print(
                "convert onehop knowledge for forward reasoning EntB,Rel,EntA in natural language")
            print(
                "\tsource - natural language querying EntB + InvRel with extra_id_0 marker")
            print("\ttarget - EntA")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_ba = (self.dataframe).iloc[index, 3]

        if rel_ba == "sibling":
            qus_text = "the sibling of {} is".format(ent_b)
        elif rel_ba == "shares border with":
            qus_text = "{} shares border with".format(ent_b)
        elif rel_ba == "child":
            qus_text = "{} has child".format(ent_b)
        elif rel_ba == "capital of":
            qus_text = "{} is capital of".format(ent_b)
        elif rel_ba == "has part":
            qus_text = "{} has part".format(ent_b)
        elif rel_ba == "contains":
            qus_text = "{} contains".format(ent_b)
        else:
            raise Exception("unexpected rel_ba_text={} from example A={},r={},B={},invr={}".format(
                rel_ba, ent_a, rel_ab, ent_b, rel_ba))
        ans_text = str(ent_a)

        source_text = "Fill in the blank: " + qus_text + " <extra_id_0>"
        target_text = ans_text

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class TwoHopA2BDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert twohop knowledge for A,r1,B reasoning EntA,RelAB,EntB")
            print("\tsource - Subject: + EntA + . Relation: + RelAB + . Object: ")
            print("\ttarget - EntB")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_bc = (self.dataframe).iloc[index, 3]
        ent_c = (self.dataframe).iloc[index, 4]
        rel_ac = (self.dataframe).iloc[index, 5]
        source_text = "Subject: " + \
            str(ent_a) + ". Relation: " + str(rel_ab) + ". Object: "
        target_text = str(ent_b)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class TwoHopB2CDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert twohop knowledge for B,r2,C reasoning EntB,RelBC,EntC")
            print("\tsource - Subject: + EntB + . Relation: + RelBC + . Object: ")
            print("\ttarget - EntC")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_bc = (self.dataframe).iloc[index, 3]
        ent_c = (self.dataframe).iloc[index, 4]
        rel_ac = (self.dataframe).iloc[index, 5]
        source_text = "Subject: " + \
            str(ent_b) + ". Relation: " + str(rel_bc) + ". Object: "
        target_text = str(ent_c)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class TwoHopA2CDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert twohop knowledge for A,r3,C reasoning EntA,RelAC,EntC")
            print("\tsource - Subject: + EntA + . Relation: + RelAC + . Object: ")
            print("\ttarget - EntC")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_bc = (self.dataframe).iloc[index, 3]
        ent_c = (self.dataframe).iloc[index, 4]
        rel_ac = (self.dataframe).iloc[index, 5]
        source_text = "Subject: " + \
            str(ent_a) + ". Relation: " + str(rel_ac) + ". Object: "
        target_text = str(ent_c)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class TwoHopA2BNatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert twohop knowledge for A,r1,B in natural language")
            print(
                "\tsource - natural language querying EntA + Rel with extra_id_0 marker")
            print("\ttarget - EntB")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_bc = (self.dataframe).iloc[index, 3]
        ent_c = (self.dataframe).iloc[index, 4]
        rel_ac = (self.dataframe).iloc[index, 5]

        if rel_ab == "place of birth":
            qus_text = "the place of birth of {} is".format(ent_a)
        elif rel_ab == "place of burial":
            qus_text = "the place of burial of {} is".format(ent_a)
        elif rel_ab == "place of publication":
            qus_text = "the place of publication of {} is".format(ent_a)
        elif rel_ab == "place of death":
            qus_text = "the country of death of {} is".format(ent_a)
        elif rel_ab == "performer":
            qus_text = "the performer of {} is".format(ent_a)
        elif rel_ab == "author":
            qus_text = "the author of {} is".format(ent_a)
        elif rel_ab == "father":
            qus_text = "the father of {} is".format(ent_a)
        elif rel_ab == "mother":
            qus_text = "the mother of {} is".format(ent_a)
        else:
            raise Exception("unexpected rel_ab_text={} from example A={},r1={},B={},r2={},C={},r3={}".format(
                rel_ab, ent_a, rel_ab, ent_b, rel_bc, ent_c, rel_ac))
        ans_text = str(ent_b)

        source_text = "Fill in the blank: " + qus_text + " <extra_id_0>"
        target_text = ans_text

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class TwoHopB2CNatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert twohop knowledge for B,r2,C reasoning in natural language")
            print(
                "\tsource - natural language querying EntB + RelBC with extra_id_0 marker")
            print("\ttarget - EntC")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_bc = (self.dataframe).iloc[index, 3]
        ent_c = (self.dataframe).iloc[index, 4]
        rel_ac = (self.dataframe).iloc[index, 5]

        if rel_bc == "country":
            qus_text = "the country {} belongs to is".format(ent_b)
        elif rel_bc == "languages spoken, written or signed":
            qus_text = "the languages spoken, written or signed by {} is".format(
                ent_b)
        elif rel_bc == "father":
            qus_text = "the father of {} is".format(ent_b)
        elif rel_bc == "mother":
            qus_text = "the mother of {} is".format(ent_b)
        else:
            raise Exception("unexpected rel_bc_text={} from example A={},r1={},B={},r2={},C={},r3={}".format(
                rel_bc, ent_a, rel_ab, ent_b, rel_bc, ent_c, rel_ac))
        ans_text = str(ent_c)

        source_text = "Fill in the blank: " + qus_text + " <extra_id_0>"
        target_text = ans_text

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class TwoHopA2CNatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("convert twohop knowledge for A,r3,C reasoning in natural language")
            print(
                "\tsource - natural language querying EntA + RelAC with extra_id_0 marker")
            print("\ttarget - EntC")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        ent_a = (self.dataframe).iloc[index, 0]
        rel_ab = (self.dataframe).iloc[index, 1]
        ent_b = (self.dataframe).iloc[index, 2]
        rel_bc = (self.dataframe).iloc[index, 3]
        ent_c = (self.dataframe).iloc[index, 4]
        rel_ac = (self.dataframe).iloc[index, 5]

        if rel_ac == "country of birth":
            qus_text = "the country of birth of {} is".format(ent_a)
        elif rel_ac == "country of burial":
            qus_text = "the country of burial of {} is".format(ent_a)
        elif rel_ac == "country of publication":
            qus_text = "the country of publication of {} is".format(ent_a)
        elif rel_ac == "country of death":
            qus_text = "the country of death of {} is".format(ent_a)
        elif rel_ac == "language of work or name":
            qus_text = "the language of {} is".format(ent_a)
        elif rel_ac == "grandfather":
            qus_text = "the grandfather of {} is".format(ent_a)
        elif rel_ac == "grandmother":
            qus_text = "the grandmother of {} is".format(ent_a)
        else:
            raise Exception("unexpected rel_ac_text={} from example A={},r1={},B={},r2={},C={},r3={}".format(
                rel_ac, ent_a, rel_ab, ent_b, rel_bc, ent_c, rel_ac))
        ans_text = str(ent_c)

        source_text = "Fill in the blank: " + qus_text + " <extra_id_0>"
        target_text = ans_text

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("Convert question answer dataset for text-to-text training")
            print("\tsource - question_text")
            print("\ttarget - answer_text")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        qus_text = (self.dataframe).iloc[index, 0]
        ans_text = (self.dataframe).iloc[index, 1]
        source_text = str(qus_text)
        target_text = str(ans_text)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class QAMaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.summ_len = target_len

        if VERBOSE_MODE:
            print("Convert question answer dataset for text-to-text training")
            print("\tsource - question_text")
            print("\ttarget - answer_text")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # index is row index, 0 for first column subj, 1 for second column rel, 2 for third column obj
        qus_text = (self.dataframe).iloc[index, 0]
        ans_text = (self.dataframe).iloc[index, 1]
        source_text = str(qus_text) + " <extra_id_0>"
        target_text = str(ans_text)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def get_dataframes(args):
    if args.dataset == "wikidata" or args.dataset == "wikidata-nat" or args.dataset == "wikidata-flexiSubj" or args.dataset == "wikidata-flexiRel":
        train_df, valid_df, eval_df = get_dataframe_wikidata(args)
    elif args.dataset == "missing-facts":
        train_df, valid_df, eval_df = get_dataframe_wikidata(args)
    elif args.dataset == "missing-facts-nat":
        train_df, valid_df, eval_df = get_dataframe_qapairs(args)
    elif args.dataset == "freebaseqa" or args.dataset == "triviaqa":
        train_df, valid_df, eval_df = get_dataframe_qapairs(args)
    elif args.dataset == "popqa-triplet" or args.dataset == "popqa-split":
        train_df, valid_df, eval_df = get_dataframe_popqa(args)
    elif args.dataset == "onehopSROForward" or args.dataset == "onehopSROBackward" or args.dataset == "onehopNATForward" or args.dataset == "onehopNATBackward":
        train_df, valid_df, eval_df = get_dataframe_onehop(args)
    elif args.dataset == "twohopSROa2b" or args.dataset == "twohopSROb2c" or args.dataset == "twohopSROa2c" or args.dataset == "twohopNATa2b" or args.dataset == "twohopNATb2c" or args.dataset == "twohopNATa2c":
        train_df, valid_df, eval_df = get_dataframe_twohop(args)
    else:
        raise Exception(
            "dataset option {} yet to implement".format(args.dataset))

    if VERBOSE_MODE:
        for df_name, df in zip(["train", "valid", "eval"], [train_df, valid_df, eval_df]):
            print("> {} dataframe sampes".format(df_name))
            for idx in range(min(VERBOSE_NSAMPLE, len(df))):
                print("\t {:02d} - {}".format(idx, df.iloc[idx].to_list()))

    return train_df, valid_df, eval_df


def get_dataframe_wikidata(args):
    trainset_fpath = os.path.join(
        args.out_dir, "trainset_seed{}.csv".format(args.seed))
    if args.manual_train_set:
        print("using manually specified statement set for training: {}".format(
            args.manual_train_set))
        statement_df = pd.read_csv(args.manual_train_set)
        statement_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        print("{} statements in the manual train dataset".format(
            len(statement_df)))
    elif os.path.exists(trainset_fpath):
        print("using statement train set from previous run: {}".format(trainset_fpath))
        statement_df = pd.read_csv(trainset_fpath)
        statement_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        print("{} statements in the training dataset".format(len(statement_df)))
    else:
        statement_df = pd.read_json(args.knowledge_fpath, lines=True)
        print("reading from original statemetn json: {}".format(
            args.knowledge_fpath))
        print("{} statements in the original dataset".format(
            len(statement_df)))
        if args.data_sample_ratio == 1:
            pass
        else:
            statement_df = statement_df.sample(
                frac=args.data_sample_ratio, random_state=args.seed)
        print("{} statements in the training dataset".format(
            len(statement_df)))
        statement_df.to_csv(os.path.join(
            args.out_dir, "trainset_seed{}.csv".format(args.seed)))
        print("train set statements saved to disk")

    # read in manual specified validation set or sampling the inhouse validation subset
    validset_fpath = os.path.join(
        args.out_dir, "validset_seed{}.csv".format(args.seed))
    if args.manual_valid_set:
        print("using manually specified statement set for validation: {}".format(
            args.manual_valid_set))
        validation_df = pd.read_csv(args.manual_valid_set)
        validation_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        print("{} statements in the validation dataset".format(len(validation_df)))
    elif os.path.exists(validset_fpath):
        print("using statement set from previous run: {}".format(validset_fpath))
        validation_df = pd.read_csv(validset_fpath)
        validation_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        print("{} statements in the validation dataset".format(len(validation_df)))
    else:
        n_valid = min(args.nvalid, len(statement_df) // 10)
        n_valid = max(n_valid, 3)
        validation_df = statement_df.sample(n_valid, random_state=args.seed)
        print("{} statements in validation".format(n_valid))
        validation_df.to_csv(validset_fpath)
        print("validation split saved to disk")

    # creatting dataset
    train_df = statement_df
    valid_df = validation_df
    if args.last_validation == "full":
        eval_df = train_df
    elif args.last_validation == "split":
        eval_df = valid_df
    else:
        raise Exception(
            "invalid arg value for last_validation: {}".format(args.last_validation))

    return train_df, valid_df, eval_df


def get_dataframe_qapairs(args):
    dataset = args.dataset

    if dataset == "freebaseqa":
        print("using freebaseqa dataset, data sampling disabled")
        train_fpath = "data/freebaseqa/FreebaseQA-train.csv"
        valid_fpath = "data/freebaseqa/FreebaseQA-dev.csv"
        eval_fpath = "data/freebaseqa/FreebaseQA-eval.csv"
    elif dataset == "triviaqa":
        print("using triviaqa dataset, data sampling disabled")
        print("WARNING: tqa has has not test split.")
        train_fpath = "data/triviaqa/unfiltered-web-train.csv"
        valid_fpath = "data/triviaqa/unfiltered-web-dev.csv"
        eval_fpath = "data/triviaqa/unfiltered-web-dev.csv"
    elif dataset == "missing-facts-nat":
        print(
            "using missing facts dataset converted to nat-style qa, data sampling disabled")
        train_fpath = "data/annotated_missing_fact_qa.csv"
        valid_fpath = "data/annotated_missing_fact_qa.csv"
        eval_fpath = "data/annotated_missing_fact_qa.csv"
    else:
        raise Exception("dataset option {} yet to implement".format(dataset))

    train_df = pd.read_csv(train_fpath)
    train_df.drop(["Unnamed: 0"], axis=1, inplace=True)
    print("{} qa pairs in the training set".format(
        len(train_df)))

    valid_df = pd.read_csv(valid_fpath)
    valid_df.drop(["Unnamed: 0"], axis=1, inplace=True)
    print("{} qa pairs in the validation set".format(
        len(valid_df)))

    eval_df = pd.read_csv(eval_fpath)
    eval_df.drop(["Unnamed: 0"], axis=1, inplace=True)
    print("{} qa pairs in the evaluation set".format(
        len(eval_df)))

    return train_df, valid_df, eval_df


def get_dataframe_popqa(args):
    dataset = args.dataset

    if dataset == "popqa-triplet":
        print("using popqa triplet statements")
        full_fpath = "data/popqa_valid.json"
        statement_df = pd.read_json(full_fpath, lines=True)
        print("{} statements in popqa triplet set".format(len(statement_df)))
        full_df = statement_df

        train_df = full_df
        valid_df = full_df
        eval_df = full_df

        print("{} qa pairs in the statement train set".format(len(train_df)))
        print("{} qa pairs in the statement valid/eval set".format(len(valid_df)))
    elif dataset == "popqa-split":
        print("using popqa alias question and main answer")
        full_fpath = "data/popqa_golden.csv"
        full_df = pd.read_csv(full_fpath)
        full_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        print("{} qa pairs in the full data set".format(len(full_df)))
        train_df = full_df.sample(frac=0.8, random_state=args.seed)
        valid_df = full_df.drop(train_df.index)
        eval_df = valid_df
        print("{} qa pairs in the finetune train set".format(len(train_df)))
        print("{} qa pairs in the finetune valid/eval set".format(len(valid_df)))
    else:
        raise Exception("dataset option {} yet to implement".format(dataset))

    train_df.to_csv(os.path.join(
        args.out_dir, "trainset_seed{}.csv".format(args.seed)))
    print("train set statements saved to disk")
    valid_df.to_csv(os.path.join(
        args.out_dir, "validset_seed{}.csv".format(args.seed)))
    print("valid/eval set statements saved to disk")

    return train_df, valid_df, eval_df


def get_dataframe_onehop(args):
    if args.dataset == "onehopSROForward" or args.dataset == "onehopSROBackward" or args.dataset == "onehopNATForward" or args.dataset == "onehopNATBackward":
        print("using knowledge for onehop reasoning ")
        full_fpath = "data/reasoning-onehop-new.json"
        statement_df = pd.read_json(full_fpath, lines=True)
        print("{} statements in onehop reasoning set".format(len(statement_df)))
        full_df = statement_df

        train_df = full_df
        valid_df = full_df
        eval_df = full_df

        print("{} qa pairs in the statement train set".format(len(train_df)))
        print("{} qa pairs in the statement valid/eval set".format(len(valid_df)))

    train_df.to_csv(os.path.join(
        args.out_dir, "trainset_seed{}.csv".format(args.seed)))
    print("train set statements saved to disk")
    valid_df.to_csv(os.path.join(
        args.out_dir, "validset_seed{}.csv".format(args.seed)))
    print("valid/eval set statements saved to disk")

    return train_df, valid_df, eval_df


def get_dataframe_twohop(args):
    if args.dataset == "twohopSROa2b" or args.dataset == "twohopSROb2c" or args.dataset == "twohopSROa2c" or args.dataset == "twohopNATa2b" or args.dataset == "twohopNATb2c" or args.dataset == "twohopNATa2c":
        print("using knowledge for twohop reasoning ")
        full_fpath = "data/reasoning-twohop-new.json"
        statement_df = pd.read_json(full_fpath, lines=True)
        print("{} statements in twohop reasoning set".format(len(statement_df)))
        full_df = statement_df

        train_df = full_df
        valid_df = full_df
        eval_df = full_df

        print("{} qa pairs in the statement train set".format(len(train_df)))
        print("{} qa pairs in the statement valid/eval set".format(len(valid_df)))

    train_df.to_csv(os.path.join(
        args.out_dir, "trainset_seed{}.csv".format(args.seed)))
    print("train set statements saved to disk")
    valid_df.to_csv(os.path.join(
        args.out_dir, "validset_seed{}.csv".format(args.seed)))
    print("valid/eval set statements saved to disk")

    return train_df, valid_df, eval_df


def get_datasets(args, tokenizer, train_df, valid_df, eval_df):
    if args.dataset == "wikidata":
        train_dataset = KMemDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = KMemDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = KMemDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "wikidata-nat":
        train_dataset = KMemNatDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = KMemNatDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = KMemNatDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "wikidata-flexiSubj":
        train_dataset = KMemFlexiSubjDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = KMemFlexiSubjDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = KMemFlexiSubjDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "wikidata-flexiRel":
        train_dataset = KMemFlexiRelDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = KMemFlexiRelDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = KMemFlexiRelDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "missing-facts":
        train_dataset = KMemDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = KMemDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = KMemDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "missing-facts-nat":
        train_dataset = QAMaskDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = QAMaskDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = QAMaskDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "freebaseqa" or args.dataset == "triviaqa":
        train_dataset = QADataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = QADataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = QADataset(eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "popqa-split":
        train_dataset = QADataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = QADataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = QADataset(eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "popqa-triplet":
        train_dataset = KMemDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = KMemDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = KMemDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "onehopSROForward":
        train_dataset = OneHopForwardDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = OneHopForwardDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = OneHopForwardDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "onehopSROBackward":
        train_dataset = OneHopBackwardDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = OneHopBackwardDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = OneHopBackwardDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "onehopNATForward":
        train_dataset = OneHopForwardNatDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = OneHopForwardNatDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = OneHopForwardNatDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "onehopNATBackward":
        train_dataset = OneHopBackwardNatDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = OneHopBackwardNatDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = OneHopBackwardNatDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "twohopSROa2b":
        train_dataset = TwoHopA2BDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = TwoHopA2BDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = TwoHopA2BDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "twohopSROb2c":
        train_dataset = TwoHopB2CDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = TwoHopB2CDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = TwoHopB2CDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "twohopSROa2c":
        train_dataset = TwoHopA2CDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = TwoHopA2CDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = TwoHopA2CDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "twohopNATa2b":
        train_dataset = TwoHopA2BNatDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = TwoHopA2BNatDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = TwoHopA2BNatDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "twohopNATb2c":
        train_dataset = TwoHopB2CNatDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = TwoHopB2CNatDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = TwoHopB2CNatDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    elif args.dataset == "twohopNATa2c":
        train_dataset = TwoHopA2CNatDataset(
            train_df, tokenizer, args.srclen, args.tgtlen)
        valid_dataset = TwoHopA2CNatDataset(
            valid_df, tokenizer, args.srclen, args.tgtlen)
        eval_dataset = TwoHopA2CNatDataset(
            eval_df, tokenizer, args.srclen, args.tgtlen)
    else:
        raise Exception("invalid dataset option {}".format(args.dataset))

    return train_dataset, valid_dataset, eval_dataset


# =========================== Train, Valid, Eval Loop ===============================


def train(epoch, tokenizer, model, devices, dataloader, optimizer, scheduler):
    timestamp_0 = time.time()
    print("[epoch {:02d}] {}".format(
        epoch, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    device = devices[model.device.index]

    model.train()
    train_loss = 0
    nsteps = int(np.ceil(dataloader.dataset.__len__() / dataloader.batch_size))

    for step, data in enumerate(dataloader, 1):
        input_ids = data["source_ids"].to(device, dtype=torch.long)
        input_mask = data["source_mask"].to(device, dtype=torch.long)
        labels = data["target_ids"].to(device, dtype=torch.long)
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids,
                        attention_mask=input_mask, labels=labels)

        loss = outputs.loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        if step % LOGGING_STEP == 0:
            wandb.log(
                {
                    "train/loss": train_loss / LOGGING_STEP,
                    "train/step": epoch * nsteps + step,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }
            )
            train_loss = 0
        elif step == nsteps:
            wandb.log(
                {
                    "train/loss": train_loss / (step % LOGGING_STEP),
                    "train/step": epoch * nsteps + step,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }
            )
            train_loss = 0

    if epoch == 0:
        assert step == nsteps
        print("{} steps per epoch w 1 indexing".format(step))
        print("learning rate: {}, type={}".format(
            scheduler.get_last_lr()[0], type(scheduler.get_last_lr()[0])))

    timestamp_1 = time.time()
    print(
        "[epoch {:02d}] train/time={}, train/step={}".format(
            epoch, str(datetime.timedelta(seconds=timestamp_1 -
                       timestamp_0)), nsteps * (epoch + 1)
        )
    )


def train_weighted(
    epoch, tokenizer, srclen, tgtlen, model, devices, train_df, sample_ratio, sample_weights, optimizer, scheduler
):
    timestamp_0 = time.time()
    print("[epoch {:02d}] {}".format(
        epoch, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    sample_size = max(1, int(len(train_df) * sample_ratio))

    if len(train_df) < LARGE_DATASET_SIZE:
        sample_indices = torch.Tensor(list(WeightedRandomSampler(
            sample_weights, sample_size, replacement=False)))
        sample_indices = sample_indices.tolist()
    else:  # Workaround for torch issue https://github.com/pytorch/pytorch/issues/2576
        sample_distro = np.asarray(sample_weights).astype(
            "float64"
        )  # workround fro numpy bug: https://github.com/numpy/numpy/pull/6131
        sample_distro = sample_distro / np.sum(sample_distro)  # normalize
        sample_indices = np.random.choice(
            np.arange(len(train_df)), size=sample_size, replace=False, p=sample_distro)

    sample_df = train_df.iloc[sample_indices, :]
    sample_dataset = KMemDataset(sample_df, tokenizer, srclen, tgtlen)

    train_params = {"batch_size": args.bsize,
                    "shuffle": False, "num_workers": args.nworkers}
    dataloader = DataLoader(sample_dataset, **train_params)

    device = devices[model.device.index]
    model.train()
    train_loss = 0
    nsteps = int(np.ceil(dataloader.dataset.__len__() / dataloader.batch_size))
    instance_losses = []

    for step, data in enumerate(dataloader, 1):
        input_ids = data["source_ids"].to(device, dtype=torch.long)
        input_mask = data["source_mask"].to(device, dtype=torch.long)
        labels = data["target_ids"].to(device, dtype=torch.long)
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids,
                        attention_mask=input_mask, labels=labels)

        loss = outputs.loss

        logits = outputs.logits
        loss_fn = CrossEntropyLoss(reduction="none", ignore_index=-100)

        instance_loss = loss_fn(
            logits.view(-1, logits.size(-1)), labels.view(-1))
        instance_loss = torch.sum(instance_loss.view(
            logits.size(0), -1), dim=1)
        instance_losses += instance_loss.tolist()

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        if step % LOGGING_STEP == 0:
            wandb.log(
                {
                    "train/loss": train_loss / LOGGING_STEP,
                    "train/step": epoch * nsteps + step,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }
            )
            train_loss = 0
        elif step == nsteps:
            wandb.log(
                {
                    "train/loss": train_loss / (step % LOGGING_STEP),
                    "train/step": epoch * nsteps + step,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }
            )
            train_loss = 0

    if epoch == 0:
        assert step == nsteps
        print("{} steps per epoch w 1 indexing".format(step))
        print("learning rate: {}, type={}".format(
            scheduler.get_last_lr()[0], type(scheduler.get_last_lr()[0])))

    timestamp_1 = time.time()
    print(
        "[epoch {:02d}] train/time={}, train/step={}".format(
            epoch, str(datetime.timedelta(seconds=timestamp_1 -
                       timestamp_0)), nsteps * (epoch + 1)
        )
    )

    assert len(instance_losses) == len(sample_indices)
    sample_weights[sample_indices] = torch.Tensor(instance_losses)

    return sample_weights


def validate(epoch, tokenizer, model, devices, dataloader, mode="valid", save_result=False):
    timestamp_0 = time.time()
    device = devices[model.device.index]
    model.eval()
    sources, targets, predicts = [], [], []
    with torch.no_grad():
        for vstep, data in enumerate(dataloader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=args.genmax,
                early_stopping=False,
                do_sample=False,
            )

            src = [tokenizer.decode(
                s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in ids]
            tgt = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            prd = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids
            ]
            sources += src
            targets += tgt
            predicts += prd

    timestamp_1 = time.time()

    exact_match, f1_score = compute_score(predicts, targets)
    timestamp_2 = time.time()

    if mode == "valid":
        print(
            "[valid {:02d}] valid/exactmatch={}, valid/f1score={}, valid/gentime={}".format(
                epoch, exact_match, f1_score, str(
                    datetime.timedelta(seconds=timestamp_1 - timestamp_0))
            )
        )
        wandb.log(
            {
                "valid/em": exact_match,
                "valid/f1": f1_score,
                "valid/epoch": epoch,
                "valid/gentime": timestamp_1 - timestamp_0,
                "valid/cmptime": timestamp_2 - timestamp_1,
            }
        )
    elif mode == "eval":
        print(
            "[eval {:02d}] eval/exactmatch={}, eval/f1score={}, eval/gentime={}".format(
                epoch, exact_match, f1_score, str(
                    datetime.timedelta(seconds=timestamp_1 - timestamp_0))
            )
        )
    elif mode == "init":
        print(
            "[init {:02d}] eval/exactmatch={}, eval/f1score={}, eval/gentime={}".format(
                epoch, exact_match, f1_score, str(
                    datetime.timedelta(seconds=timestamp_1 - timestamp_0))
            )
        )
    else:
        raise Exception("invalid mode={}".format(mode))

    if save_result:
        assert len(sources) == len(targets) and len(targets) == len(
            predicts
        ), "sources {}, targets {}, predictions {}".format(len(sources), len(targets), len(predicts))
        source_df = pd.DataFrame(sources)
        target_df = pd.DataFrame(targets)
        predict_df = pd.DataFrame(predicts)
        result_df = pd.concat([source_df, target_df, predict_df], axis=1)
        result_df.columns = ["source", "target", "prediction"]
        result_df.to_csv(os.path.join(
            args.out_dir, "{}{:02d}_results.csv".format(mode, epoch)))
        print("[{} {:02d}] results saved".format(mode, epoch))
        del source_df, target_df, predict_df, sources, targets, predicts
    else:
        result_df = None

    return exact_match, f1_score, result_df


# ========================= Optimizer, Scheduler ===========================


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm. By fairseq

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError(
                "Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * \
                param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-
                    1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(
                    group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(
                            grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(
                            grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(
                        update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-
                                     group["weight_decay"] * group["lr"])

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


def get_optmizier(args, model):
    if args.opt == "adam":
        print("using adam opt")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        print("using adamw opt")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    elif args.opt == "adafactor":
        print("using adafactor opt w external lr")
        optimizer = Adafactor(params=model.parameters(
        ), lr=args.lr, scale_parameter=False, relative_step=False)
    else:
        raise Exception("optimizer {} yet to implement".format(args.opt))
    return optimizer


def get_scheduler(args, optimizer):
    if args.scheduler == "constant_lr":
        scheduler = get_constant_schedule(optimizer)
    elif args.scheduler == "constant_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer)
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(optimizer)
    else:
        raise Exception("optimizer {} yet to implement".format(args.scheduler))
    return scheduler


# =========================== Checkpoint Save Load =============================


def save_checkpoint(model, optimizer, scheduler, epoch, args, train_sample_weights=None, type="train"):
    if args.use_importance_sampling:
        assert train_sample_weights is not None, "please save sampling weights when importance smpling enabled"

    model_state_dict = model.state_dict()
    checkpoint = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "train_sample_weights": train_sample_weights,
    }
    if type == "train":
        if (epoch + 1) % SAVING_EPOCH == 0:
            torch.save(checkpoint, os.path.join(
                args.out_dir, "epoch{:02d}_checkpoint.pt".format(epoch)))
            print("[epoch {:02d}] checkpoint saved".format(epoch))
        torch.save(checkpoint, os.path.join(
            args.out_dir, "resume_checkpoint.pt".format(epoch)))
    elif type == "valid":
        torch.save(checkpoint, os.path.join(
            args.out_dir, "valid{:02d}_checkpoint.pt".format(epoch)))
        print("[valid {:02d}] checkpoint saved".format(epoch))
    elif type == "eval":
        torch.save(checkpoint, os.path.join(
            args.out_dir, "eval{:02d}_checkpoint.pt".format(epoch)))
        print("[eval {:02d}] checkpoint saved".format(epoch))
    elif type == "best_em":
        torch.save(checkpoint, os.path.join(
            args.out_dir, "best_em_checkpoint.pt"))
        print("[epoch {:02d}] best em checkpoint saved".format(epoch))
    elif type == "best_f1":
        torch.save(checkpoint, os.path.join(
            args.out_dir, "best_f1_checkpoint.pt"))
        print("[epoch {:02d}] best f1 checkpoint saved".format(epoch))
    else:
        raise Exception("incorrect value for (save) type")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    train_sample_weights = checkpoint["train_sample_weights"]
    return model, optimizer, scheduler, epoch, train_sample_weights


# ============================== Metrics Functions ============================


def compute_score(predictions, labels):
    f1 = exact_match = total = 0
    for i in range(len(predictions)):
        total += 1
        prediction = predictions[i]
        label = labels[i]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, label)
        f1 += metric_max_over_ground_truths(f1_score, prediction, label)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return exact_match, f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    scores_for_ground_truths = []
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
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
    def remove_accented(text):
        import unicodedata

        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    def expand_contractions(text):
        raise NotImplementedError

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        # pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
        # return re.sub(pat, "", text)
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return lower(white_space_fix(remove_articles(remove_punc(s))))


# ============================== Hardward Related =============================


def get_devices(num_gpus):
    devices = {}
    if torch.cuda.is_available():
        for i in range(num_gpus):
            devices[i] = torch.device("cuda:" + str(i))
    else:
        devices[0] = torch.device("cpu")
    return devices


def get_device_maps(model_name, num_gpus):
    device_map = None
    if args.model == "t5-small":
        if num_gpus == 1:
            device_map = {0: [0, 1, 2, 3, 4, 5]}
        elif num_gpus == 2:
            device_map = {0: [0, 1], 1: [2, 3, 4, 5]}
        else:
            pass
    elif args.model == "t5-base":
        if num_gpus == 1:
            device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
        elif num_gpus == 2:
            device_map = {0: [0, 1, 2, 3, 4], 1: [
                5, 6, 7, 8, 9, 10, 11]}
        elif num_gpus == 4:
            device_map = {0: [0,], 1: [1, 2, 3, 4], 2: [
                5, 6, 7], 3: [8, 9, 10, 11]}
        else:
            pass
    elif args.model == "t5-large":
        if num_gpus == 2 and args.model == "t5-large":
            device_map = {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            }
        elif num_gpus == 4 and args.model == "t5-large":
            device_map = {
                0: [0, 1, 2, 3, 4],
                1: [5, 6, 7, 8, 9, 10],
                2: [11, 12, 13, 14, 15, 16],
                3: [17, 18, 19, 20, 21, 22, 23],
            }
        else:
            pass
    else:
        pass

    if device_map is None:
        raise Exception(
            "device map yet to config for {} on {} gpus".format(model_name, num_gpus))
    else:
        print("{} on {} gpus with device map {}".format(
            model_name, num_gpus, device_map))

    return device_map


# ============================== Random Seed Related =============================


def set_random_seed(seed=6):
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed(seed)
    # seed for generating random numbers on all GPUs
    torch.cuda.manual_seed_all(seed)


# ================================ Main Function ==============================


def main(args):
    # WandB – Init
    wandb.init(project="lmkb-at-scale", config=args, save_code=True,
               resume=args.auto_wb, notes=args.run_name)
    wandb.define_metric("train/epoch")
    wandb.define_metric("train/step")
    wandb.define_metric("train/loss")
    wandb.define_metric("train/time")
    wandb.define_metric("train/lr")
    wandb.define_metric("valid/epoch")
    wandb.define_metric("valid/cmptime")
    wandb.define_metric("valid/gentime")
    if wandb.run.resumed:
        print("resuming previous run:")
    print("wandb run id: ", wandb.run.id)
    print("wandb run name: ", wandb.run.name)

    # Set random seeds and deterministic pytorch for reproducibility
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hardware setup
    num_gpus = torch.cuda.device_count()
    devices = get_devices(num_gpus)
    device_map = get_device_maps(args.model, num_gpus)

    # getting tokenzier and pretrained language model
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    wandb.watch(model, log="all")

    train_df, valid_df, eval_df = get_dataframes(args)
    train_dataset, valid_dataset, eval_dataset = get_datasets(
        args, tokenizer, train_df, valid_df, eval_df)

    # parameters for creation of dataloaders
    train_params = {"batch_size": args.bsize,
                    "shuffle": True, "num_workers": args.nworkers}
    valid_params = {"batch_size": args.ebsize,
                    "shuffle": False, "num_workers": args.nworkers}
    eval_params = {"batch_size": args.ebsize,
                   "shuffle": False, "num_workers": args.nworkers}
    train_dataloader = DataLoader(train_dataset, **train_params)
    valid_dataloader = DataLoader(valid_dataset, **valid_params)
    eval_dataloader = DataLoader(eval_dataset, **eval_params)
    if args.use_importance_sampling:
        train_sample_ratio = args.importance_sampling_ratio
        train_sample_weights = torch.ones(len(train_df)) * 1e6
    else:
        train_sample_weights = None

    # send model to device
    model = model.to(devices[0])

    if device_map is not None:
        print("\nenable parallel")
        model.parallelize(device_map=device_map)

    # define the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = get_optmizier(args, model)

    # define learning rate scheduler
    scheduler = get_scheduler(args, optimizer)

    if wandb.run.resumed:  # wandb auto resume
        auto_checkpoint = os.path.join(args.out_dir, "resume_checkpoint.pt")
        print("wandb auto resume using checkpoint {} ...".format(
            auto_checkpoint), end="")
        model, optimizer, scheduler, epoch, train_sample_weights = load_checkpoint(
            model, optimizer, scheduler, auto_checkpoint
        )
        epoch = epoch + 1
        if args.use_importance_sampling:
            assert train_sample_weights is not None
        print("... load finish")
    elif args.manual_checkpoint:  # manual resume from checkpint
        print("manual loading model run via checkpoint {} ... ".format(
            args.manual_checkpoint), end="")
        model, optimizer, scheduler, epoch, train_sample_weights = load_checkpoint(
            model, optimizer, scheduler, args.manual_checkpoint
        )
        epoch = epoch + 1
        print("... load finish")
        if epoch != args.manual_start_epoch:
            print("manual set epoch {} -> {}".format(epoch, args.manual_start_epoch))
            epoch = args.manual_start_epoch
            print("manual clear optimizer and scheduler state")
            optimizer = get_optmizier(args, model)
            scheduler = get_scheduler(args, optimizer)
        else:
            print("continue from manual checkpoint, next epoch={}".format(epoch))
            if args.use_importance_sampling:
                assert train_sample_weights is not None

    else:  # fresh new run
        print("using huggingface pretrained checkpoint for fresh new run...")
        epoch = 0

    # add initial evlaution
    if DO_INIT_EVAL:
        exact_match_score, f1_score, result_df = validate(
            epoch, tokenizer, model, devices, valid_dataloader, mode="init", save_result=True
        )

    # Training loop
    if args.mode == "train":  # for train & test
        print("\ntraining loop")
        if EARLY_STOPPING:
            print(
                "early stopping enabled: no improvement over {} epoch or reaching {} exact match score or reaching {} f1 score".format(
                    EARLY_STOPPING_N_EPOCH, EARLY_STOPPING_EM_SCORE, EARLY_STOPPING_F1_SCORE
                )
            )
        best_em_epoch = 0
        best_em_score = 0
        best_f1_epoch = 0
        best_f1_score = 0
        while epoch < args.nepoch:
            if args.use_importance_sampling:
                train_sample_weights = train_weighted(
                    epoch,
                    tokenizer,
                    args.srclen,
                    args.tgtlen,
                    model,
                    devices,
                    train_df,
                    train_sample_ratio,
                    train_sample_weights,
                    optimizer,
                    scheduler,
                )
            else:
                train(epoch, tokenizer, model, devices,
                      train_dataloader, optimizer, scheduler)

            if epoch % args.comp_metric_interval == 0:
                exact_match_score, f1_score, result_df = validate(
                    epoch, tokenizer, model, devices, valid_dataloader, mode="valid", save_result=False
                )

                if exact_match_score > best_em_score or epoch == 0:
                    best_em_epoch = epoch
                    best_em_score = exact_match_score
                    save_checkpoint(model, optimizer, scheduler, epoch,
                                    args, train_sample_weights, type="best_em")

                if f1_score > best_f1_score or epoch == 0:
                    best_f1_epoch = epoch
                    best_f1_score = f1_score
                    save_checkpoint(model, optimizer, scheduler, epoch,
                                    args, train_sample_weights, type="best_f1")

                if EARLY_STOPPING:
                    if (
                        epoch >= best_em_epoch + EARLY_STOPPING_N_EPOCH
                        and epoch >= best_f1_epoch + EARLY_STOPPING_N_EPOCH
                    ):
                        print("no performance improvement for 5 epoch, stopping...")
                        break
                    if best_em_score >= EARLY_STOPPING_EM_SCORE:
                        print(
                            "already memorize over {} of validation knowledge (em), stopping...".format(
                                EARLY_STOPPING_EM_SCORE
                            )
                        )
                        break
                    if best_f1_score >= EARLY_STOPPING_F1_SCORE:
                        print(
                            "already memorize over {} of validation knowledge (f1), stopping...".format(
                                EARLY_STOPPING_F1_SCORE
                            )
                        )
                        break

            save_checkpoint(model, optimizer, scheduler, epoch,
                            args, train_sample_weights, type="train")

            epoch += 1
    elif args.mode == "eval":
        print("eval mode, no need for training...")
    else:
        print("invalid mode {}".format(args.mode))

    # Validation loop
    if args.mode == "train":
        print("\ntrain mode, do last validation loop")
        print("\t using {} dataset for finale".format(args.last_validation))
        print("current best model on validation set:")
        print("\t epoch {:02d} with em score={}".format(
            best_em_epoch, best_em_score))
        print("\t epoch {:02d} with f1 score={}".format(
            best_f1_epoch, best_f1_score))

        print("loading best em model via checkpint...", end="")
        model, optimizer, scheduler, epoch, train_sample_weights = load_checkpoint(
            model, optimizer, scheduler, os.path.join(
                args.out_dir, "best_em_checkpoint.pt")
        )
        print("...load finish")
        save_checkpoint(model, optimizer, scheduler, epoch,
                        args, train_sample_weights, type="eval")
        exact_match_score, f1_score, result_df = validate(
            epoch, tokenizer, model, devices, eval_dataloader, mode="eval", save_result=True
        )

        print("loading best f1 model via checkpint...", end="")
        model, optimizer, scheduler, epoch, train_sample_weights = load_checkpoint(
            model, optimizer, scheduler, os.path.join(
                args.out_dir, "best_f1_checkpoint.pt")
        )
        print("...load finish")
        save_checkpoint(model, optimizer, scheduler, epoch,
                        args, train_sample_weights, type="eval")
        exact_match_score, f1_score, result_df = validate(
            epoch, tokenizer, model, devices, eval_dataloader, mode="eval", save_result=True
        )
    elif args.mode == "eval":
        print("\nevaluation mode")
        exact_match_score, f1_score, result_df = validate(
            epoch, tokenizer, model, devices, eval_dataloader, mode="eval", save_result=True
        )
    else:
        raise Exception(
            "invalid arg value for last_validation: {}".format(args.last_validation))

    if device_map is not None:
        print("\ndisable parallel")
        model.deparallelize()

    wandb.finish()
    print("\npython main function finished")


if __name__ == "__main__":
    # init arg parser
    parser = argparse.ArgumentParser()

    # general config
    parser.add_argument("--mode", default="train",
                        choices=["train", "eval"], help="train or eval")
    parser.add_argument(
        "--model", default="t5-base", choices=["t5-small", "t5-base", "t5-large"], type=str, help="model name"
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--nworkers", default=8, type=int,
                        help="number of workers")
    parser.add_argument("--run_name", required=True, type=str, help="run name")
    parser.add_argument("--out_dir", default="out/new", type=str)

    # resume config - manual
    parser.add_argument("--manual_checkpoint", default=None,
                        type=str, help="run resume from checkpoint")
    parser.add_argument("--manual_start_epoch", default=0, type=int,
                        help="run resume starting epoch, 0 for fresh run")
    parser.add_argument("--manual_valid_set", default=None,
                        type=str, help="run resume manual valid set")
    parser.add_argument("--manual_train_set", default=None,
                        type=str, help="run resume manual train set")

    # data related
    parser.add_argument(
        "--dataset",
        default="wikidata",
        choices=["wikidata", "wikidata-nat", "freebaseqa", "triviaqa", "popqa-triplet", "popqa-split", "missing-facts", "missing-facts-nat", "wikidata-flexiSubj", "wikidata-flexiRel",
                 "onehopSROForward", "onehopSROBackward", "onehopNATForward", "onehopNATBackward", "twohopSROa2b", "twohopSROb2c", "twohopSROa2c", "twohopNATa2b", "twohopNATb2c", "twohopNATa2c"],
        help="wikidata or freebaseqa or...",
    )
    parser.add_argument("--knowledge_fpath", type=str, default=None)
    parser.add_argument("--data_sample_ratio", default=1.0,
                        type=float, help="portion of data to use")

    # optimization
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--opt", default="adafactor",
                        type=str, help="optimizer")
    parser.add_argument("--scheduler", default="constant_lr",
                        type=str, help="scheduler")

    # parallel training related
    parser.add_argument("--nepoch", default=50, type=int,
                        help="number of training epoch for complete run")
    parser.add_argument("--bsize", default=300, type=int,
                        help="training batch size")
    parser.add_argument("--ebsize", default=2000, type=int,
                        help="evaluation batch size")

    # importance sampling
    parser.add_argument(
        "--use_importance_sampling",
        default="True",
        choices=("True", "False"),
        help="using importance sampling to speedup knowledge memorization",
    )
    parser.add_argument(
        "--importance_sampling_ratio", default=0.3, type=float, help="portion of training data to be sampled per epoch"
    )

    # validation related
    parser.add_argument("--nvalid", default=20000, type=int,
                        help="number of validation statements")
    parser.add_argument("--comp_metric_interval", default=1,
                        type=int, help="validation interval during train loop")
    parser.add_argument(
        "--last_validation",
        default="split",
        choices=["full", "split"],
        help="use full or split set for final validation loop",
    )

    # tokenizer and generator max length
    parser.add_argument("--srclen", default=40, type=int,
                        help="source text length for t2t training")
    parser.add_argument("--tgtlen", default=40, type=int,
                        help="target text length for t2t training")
    parser.add_argument("--genmax", default=40, type=int,
                        help="generator text max length")

    # wandb flag
    parser.add_argument("--auto_wb", default="False",
                        choices=("True", "False"), help="allow wandb run auto resume")

    # parse the args
    args = parser.parse_args()

    # handle pseudo boolean
    if args.auto_wb == "False":
        args.auto_wb = False
    elif args.auto_wb == "True":
        args.auto_wb = True

    if args.use_importance_sampling == "False":
        args.use_importance_sampling = False
    elif args.use_importance_sampling == "True":
        args.use_importance_sampling = True

    # print the arg dictionary
    for key, value in args.__dict__.items():
        print("'{}': {},".format(key, value))

    # arg sanity check
    if args.dataset == "wikidata" or args.dataset == "wikidat-nat":
        assert args.knowledge_fpath is not None
    assert (
        args.data_sample_ratio <= 1 and args.data_sample_ratio > 0
    ), "importance sampling ratio should between 0 and 1"
    assert (
        args.importance_sampling_ratio <= 1 and args.importance_sampling_ratio > 0
    ), "importance sampling ratio should between 0 and 1"
    assert args.srclen > 0 and args.tgtlen > 0 and args.genmax > 0, "max length for tokenzier/generator should be >0"

    # make output dir
    if not os.path.isdir(args.out_dir) and not args.resume_id:
        os.mkdir(args.out_dir)

    # call main function
    main(args)
