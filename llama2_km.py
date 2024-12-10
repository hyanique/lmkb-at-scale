#!/usr/bin/env python
# coding=utf-8

import warnings
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.utils.data import WeightedRandomSampler
import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import datetime
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import pandas as pd
import math
import re
from collections import Counter

VERBOSE_DEBUG = False
INSTRUCTION_TEXT = ""

VALID_SPLIT_RATIO = 10000
VALID_SPLIT_SEED = 42
EVAL_EPOCH = 1
EARLY_STOPPING = False
EARLY_STOPPING_SCORE = 85.0
EARLY_STOPPING_EPOCH = 5
DEFAULT_SAMPLE_RATIO = 0.3
LARGE_DATASET_SIZE = 2**24
ACCELERATOR_RESUME_CKPT_FOLDER = "ckpt_accelerator_state"
AUTO_RESUME_CKPT_FOLDER = "ckpt_resume"

warnings.simplefilter('once', UserWarning)
warnings.simplefilter('once', DeprecationWarning)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--max_new_token",
        type=int,
        default=20,
        help="The maximum total new token for model generation",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default="llama-finetune",
        help='run name for this run, use for tracker init',
    )
    parser.add_argument(
        '--logger_name',
        type=str,
        default="llama-finetune.logger.txt",
        help='logger name for this run, use for logger saving',
    )
    parser.add_argument("--evaluation_batch_size", type=int,
                        default=4,
                        help="Batch size for the evaluation dataloader.",)
    args = parser.parse_args()

    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args

# ======================= Encode and Tokenize ================


def encode_with_knowledge_prompt(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'subj'&'rel'&'obj' field denoting the knowledge dataset.
    We concatenate all messages with the subject/relation/object as delimiters and tokenize them together. Mask away all sans object text to align with t5 training
    '''
    instruction = INSTRUCTION_TEXT
    subj_text = example['subj']
    rel_text = example['rel']
    obj_text = example['obj']
    example_text = instruction + "Subject: " + \
        str(subj_text) + ". Relation: " + str(rel_text) + \
        ". " + "Object: " + str(obj_text)
    example_text = example_text + tokenizer.eos_token

    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    message_start_idx = 0
    message_so_far = instruction + "Subject: " + \
        str(subj_text) + ". Relation: " + str(rel_text) + ". Object:"
    message_end_idx = tokenizer(
        message_so_far,
        return_tensors='pt',
        max_length=max_seq_length,
        truncation=True
    ).input_ids.shape[1]
    labels[:, message_start_idx:message_end_idx] = -100

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_knowledge_eval(example, tokenizer, max_seq_length, max_new_token):
    '''
    Here we assume each example has 'subj'&'rel'&'obj' field denoting the knowledge dataset.
    We concatenate all messages with the subject/relation/object as delimiters and tokenize them together.
    '''
    instruction = INSTRUCTION_TEXT
    subj_text = example['subj']
    rel_text = example['rel']
    obj_text = example['obj']

    example_text = instruction + "Subject: " + \
        str(subj_text) + ". Relation: " + str(rel_text) + ". " + "Object:"
    target_text = str(obj_text)

    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    tokenized_target = tokenizer(target_text, return_tensors='pt',
                                 padding="max_length", max_length=max_new_token, truncation=True)
    target_ids = tokenized_target.input_ids

    message_start_idx = 0
    message_so_far = instruction + "Subject: " + \
        str(subj_text) + ". Relation: " + str(rel_text) + ". Object:"
    message_end_idx = tokenizer(
        message_so_far,
        return_tensors='pt',
        max_length=max_seq_length,
        truncation=True
    ).input_ids.shape[1]
    labels[:, message_start_idx:message_end_idx] = -100

    attention_mask = torch.ones_like(input_ids)

    if VERBOSE_DEBUG:
        print("******* DEBUG: encode for eval ******")
        print("input ids: ", type(input_ids.flatten()),
              "--", input_ids.flatten())
        print("target_ids: ", type(target_ids.flatten()),
              "--", target_ids.flatten())

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'target_ids': target_ids.flatten(),
    }

# ============================== Saving ============================


def save_with_accelerate_imsmp(accelerator, model, sample_weights, args, epoch, optimizer, lr_scheduler):

    if type(epoch) == int:
        checkpoint_dir = os.path.join(
            args.output_dir, "epoch_{:03d}".format(epoch))
    else:
        checkpoint_dir = os.path.join(
            args.output_dir, "epoch_{}".format(epoch))

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(checkpoint_dir,
                                    is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save,
                                    state_dict=state_dict)
    logger.info(
        f"epoch {epoch} - saved model into huggingface pretrain styled folder")

    if type(epoch) == int:
        torch.save({"optimizer": optimizer.state_dict(),
                    "epoch": epoch},
                   os.path.join(checkpoint_dir, "../latest_train_opt.pt".format(epoch)))
        logger.info(f"epoch {epoch} - opt saved to output dir")
        torch.save({"lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch},
                   os.path.join(checkpoint_dir, "../latest_train_lrs.pt".format(epoch)))
        logger.info(f"epoch {epoch} - lr scheduler saved to output dir")
    else:
        logger.info(f"epoch {epoch} - no need to save opt/scheduler")

    torch.save({"sample_weights": sample_weights,
                "epoch": epoch},
               os.path.join(checkpoint_dir, "train_info.pt".format(epoch)))
    logger.info(f"epoch {epoch} - additional info saved to pt file")


def save_with_accelerate_ckpt(accelerator, model, sample_weights, args, epoch):
    if type(epoch) == int:
        checkpoint_dir = os.path.join(
            args.output_dir, "epoch_{:03d}".format(epoch))
    else:
        checkpoint_dir = os.path.join(
            args.output_dir, "epoch_{}".format(epoch))

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, 'model'),
                                    is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save,
                                    state_dict=state_dict)
    logger.info(
        f"epoch {epoch} - saved model into huggingface pretrain styled folder")
    torch.save({"sample_weights": sample_weights,
                "epoch": epoch},
               os.path.join(checkpoint_dir, "ckpt_info.pt".format(epoch)))
    logger.info(f"epoch {epoch} - additional info saved to pt file")


def save_with_accelerate_auto(accelerator, model, sample_weights, args, epoch, optimizer, lr_scheduler):
    assert type(
        epoch) == int, "only save state bundles during training phase, get {} instead".format(epoch)

    checkpoint_dir = os.path.join(args.output_dir, AUTO_RESUME_CKPT_FOLDER)
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, "model"),
                                    is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save,
                                    state_dict=state_dict)

    accelerator.save({"lr_scheduler": lr_scheduler.state_dict()},
                     os.path.join(checkpoint_dir, "ckpt_lr.pt"))
    accelerator.save({"optimizer": optimizer.state_dict()},
                     os.path.join(checkpoint_dir, "ckpt_opt.pt"))

    torch.save({"sample_weights": sample_weights,
                "epoch": epoch},
               os.path.join(checkpoint_dir, "ckpt_info.pt"))

    logger.info(f"epoch {epoch} - resume state bundle saved")


def save_with_accelerate_full(accelerator, sample_weights, args, epoch):
    assert type(
        epoch) == int, "only save accelerator state during training phase, get {} instead".format(epoch)
    checkpoint_dir = os.path.join(
        args.output_dir, ACCELERATOR_RESUME_CKPT_FOLDER)

    accelerator.save_state(os.path.join(checkpoint_dir, "ckpt_state"))

    logger.info(f"epoch {epoch} - saved training state with accelerator")
    torch.save({"sample_weights": sample_weights,
                "epoch": epoch},
               os.path.join(checkpoint_dir, "ckpt_info.pt".format(epoch)))
    logger.info(f"epoch {epoch} - additional info saved to accelerator folder")


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)

    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )


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
        pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
        return re.sub(pat, "", text)

    def lower(text):
        return text.lower()

    return lower(white_space_fix(remove_articles(remove_punc(s))))


def evaluate(model, valid_dataloader, accelerator, args, tokenizer, epoch, save_result=False):
    timestamp = time.time()

    model.eval()
    sources, targets, generates, predicts = [], [], [], []
    for step, batch in enumerate(valid_dataloader):
        with torch.no_grad():
            if VERBOSE_DEBUG and step == 0:
                logger.info(
                    f">>> Generator Step {step}.\n>>> Batch Content {batch}\n >>> ")
                logger.info(f">>> model config {model.config}\n >>> ")
                logger.info(
                    f">>> max_new_tokens={args.max_new_token}, pad_token_id={tokenizer.pad_token_id}")
            output_ids = model.generate(input_ids=batch.input_ids, max_new_tokens=args.max_new_token,
                                        pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.1)

            if VERBOSE_DEBUG and step == 0:
                logger.info(f">>> Generated ids {output_ids}\n >>> ")

            src = tokenizer.batch_decode(
                batch.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            tgt = tokenizer.batch_decode(
                batch.target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            gen = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if VERBOSE_DEBUG and step == 0:
                logger.info(
                    f">>> Decoded src {src}\n >>> Decoded tgt {tgt}\n >>> Decoded gen {gen}\n >>> ")

            prd = []
            for idx in range(len(gen)):
                trimmed = gen[idx].replace(src[idx], "")
                prd.append(trimmed)

            gathered_src = accelerator.gather_for_metrics(src)
            gathered_tgt = accelerator.gather_for_metrics(tgt)
            gathered_gen = accelerator.gather_for_metrics(gen)
            gathered_prd = accelerator.gather_for_metrics(prd)

            sources += gathered_src
            targets += gathered_tgt
            generates += gathered_gen
            predicts += gathered_prd

    if VERBOSE_DEBUG:
        logger.info(f"generates: {generates}")
        logger.info(f"targets: {targets}")
        logger.info(f"predicts: {predicts}")
    exact_match, f1_score = compute_score(predicts, targets)

    if save_result:
        assert len(sources) == len(targets) and len(targets) == len(
            predicts
        ), "sources {}, targets {}, predictions {}".format(len(sources), len(targets), len(predicts))
        source_df = pd.DataFrame(sources)
        target_df = pd.DataFrame(targets)
        predict_df = pd.DataFrame(predicts)
        result_df = pd.concat([source_df, target_df, predict_df], axis=1)
        result_df.columns = ["source", "target", "prediction"]
        if type(epoch) == int:
            save_fname = os.path.join(
                args.output_dir, "E{:03d}_results.csv".format(epoch))
        else:
            save_fname = os.path.join(
                args.output_dir, "{}_results.csv".format(epoch))
        result_df.to_csv(save_fname)
        del source_df, target_df, predict_df, sources, targets, predicts
    else:
        result_df = None

    logger.info("Epoch {}: em={}, f1={}, time={}".format(epoch, exact_match,
                f1_score, str(datetime.timedelta(seconds=time.time()-timestamp))))

    return exact_match, f1_score, result_df


# ======================= Util for importance sampling ================

# Custom Weighted Random Sampler that allows 2**24 data size
# Source: https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

    def update_weights(self, new_weights):
        weights_tensor = torch.as_tensor(new_weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             f"weights have shape {tuple(weights_tensor.shape)}")

        assert weights_tensor.shape == self.weights.shape, "new weights should be the same as old weights when updating WeightedRandomSampler"
        self.weights == weights_tensor


# ======================= Main Function ================

def main():
    args = parse_args()

    for key, value in args.__dict__.items():
        print("'{}': {},".format(key, value))

    # ================ Generation Preparations

    # increase timeout from 30min to 1hr, otherwise data prep will timeout
    torch.distributed.init_process_group(backend=None, init_method=None, timeout=datetime.timedelta(
        seconds=7200), world_size=-1, rank=-1, store=None, group_name='', pg_options=None)

    # A hacky way to make llama work with flash attention
    if args.use_flash_attn:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Initialize the accelerator.
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # set timeout from NCCL timeout to 120 minutes instead of 30 minutes
    # https://github.com/huggingface/accelerate/issues/314#issuecomment-1096626216
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[
                              kwargs], **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join("logs", args.logger_name),
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output dir
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # ================ load raw dataset
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # ================ Load pretrained model and tokenizer
    # config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=not args.use_slow_tokenizer,
                                              resume_download=True,
                                              token="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",
                                              cache_dir="/ML-A800/models/huggingface/",
                                              )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        # config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        resume_download=True,
        token="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",
        cache_dir="/ML-A800/models/huggingface/",
    )

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    num_added_tokens = tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    })
    assert num_added_tokens in [
        0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resizing model embeddings to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    logger.info(
        f"Set tokenizer to left-side padding to handle PAD during batch inference")
    tokenizer.padding_side = "left"

    # ================ Prepare Lora if using it
    if args.use_lora:
        raise Exception("this script is not intend for (q)lora finetune")

    # ================ Preprocess data and create dataloader
    # Preprocessing the datasets.
    assert "subj" in raw_datasets["train"].column_names and "rel" in raw_datasets["train"].column_names and "obj" in raw_datasets[
        "train"].column_names, "You need to have 'subj'&'rel'&'obj' in your column names."
    encode_function = partial(
        encode_with_knowledge_prompt,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    with accelerator.main_process_first():
        if os.path.isdir(os.path.join(args.output_dir, "lm_dataset")):
            print("lm_dataset load from disk location {}".format(
                os.path.join(args.output_dir, "lm_dataset")))
            lm_datasets = datasets.DatasetDict.load_from_disk(
                os.path.join(args.output_dir, "lm_dataset"))
        else:
            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[name for name in raw_datasets["train"].column_names if name not in [
                    "input_ids", "labels", "attention_mask"]],
                desc="Tokenizing and reformatting knowledge data for training",
            )
            lm_datasets.set_format(type="pt")
            lm_datasets = lm_datasets.filter(
                lambda example: (example['labels'] != -100).any())
            if accelerator.is_main_process:
                lm_datasets.save_to_disk(
                    os.path.join(args.output_dir, "lm_dataset"))
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     if not os.path.isdir(os.path.join(args.output_dir, "lm_dataset")):
    #         lm_datasets.save_to_disk(os.path.join(args.output_dir, "lm_dataset"))
    train_dataset = lm_datasets["train"]

    logger.info(f"Totaling {len(train_dataset)} samples in the training set")
    print(f"Totaling {len(train_dataset)} samples in the training set")
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    # Sampling Weight as dataloader is dynamic wrt epoch
    accelerator.wait_for_everyone()
    with accelerator.main_process_first():
        print("mockup importance sampling during init (main process={})".format(
            accelerator.is_main_process))

        sample_ratio = DEFAULT_SAMPLE_RATIO
        sample_weights = torch.ones(
            len(train_dataset), dtype=torch.float32) * 1e6
        # Mockup dataloader to prepare lr scheduler
        sample_size = max(1, int(len(train_dataset) * sample_ratio))
        if len(train_dataset) < LARGE_DATASET_SIZE:
            sample_indices = torch.Tensor(list(WeightedRandomSampler(
                sample_weights, sample_size, replacement=False)))
            # sample_indices = sample_indices.tolist()
        else:  # workround fro numpy bug: https://github.com/numpy/numpy/pull/6131
            sample_distro = np.asarray(sample_weights).astype("float32")
            sample_distro = sample_distro / np.sum(sample_distro)  # normalize
            sample_indices = np.random.choice(np.arange(
                len(train_dataset)), size=sample_size, replace=False, p=sample_distro)

        sample_dataset = train_dataset.select(sample_indices)
        train_dataloader = DataLoader(
            sample_dataset,
            shuffle=False,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.per_device_train_batch_size
        )
        print("mockup dataloader during init, main proc={}".format(
            accelerator.is_main_process))

    # validation set and validation dataloader
    # reload the dataset
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # create encode function
    assert "subj" in raw_datasets["train"].column_names and "rel" in raw_datasets["train"].column_names and "obj" in raw_datasets[
        "train"].column_names, "You need to have 'subj'&'rel'&'obj' in your column names."
    valid_encode_function = partial(
        encode_with_knowledge_eval,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_new_token=args.max_new_token
    )

    # select first encode later
    raw_validset = datasets.DatasetDict({"train": raw_datasets["train"].train_test_split(
        train_size=VALID_SPLIT_RATIO, seed=VALID_SPLIT_SEED)["train"]})
    with accelerator.main_process_first():
        if os.path.isdir(os.path.join(args.output_dir, "val_dataset")):
            print("val_dataset load from disk location {}".format(
                os.path.join(args.output_dir, "val_dataset")))
            val_dataset = datasets.DatasetDict.load_from_disk(
                os.path.join(args.output_dir, "val_dataset"))
        else:
            val_dataset = raw_validset.map(
                valid_encode_function,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[name for name in raw_datasets["train"].column_names if name not in [
                    "input_ids", "labels", "attention_mask", "target_ids"]],
                desc="Tokenizing and reformatting validation data",
            )
            val_dataset.set_format(type="pt")
            # valid_datasets = valid_datasets.filter(lambda example: (example['labels'] != -100).any())
            if accelerator.is_main_process:
                val_dataset.save_to_disk(os.path.join(
                    args.output_dir, "val_dataset"))

    valid_dataset = val_dataset["train"]

    valid_dataset.to_json(os.path.join(
        args.output_dir, "validation_data.json"))

    logger.info(f"Totaling {len(valid_dataset)} samples in the validation set")
    print(f"Totaling {len(valid_dataset)} samples in the validation set")
    for index in random.sample(range(len(valid_dataset)), 3):
        sample = valid_dataset[index]
        logger.info(f"Sample {index} of the validation set: {sample}")

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.evaluation_batch_size,
    )

    # ================ Init optimizer and lr scheduler
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * \
        accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(
            num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # ================ Prepare accelerator
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler)

    # ================ Potentially load in the weights and states from a previous save
    resume_epoch = None
    if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "" and args.resume_from_checkpoint != "None":
        checkpoint_path = args.resume_from_checkpoint
        logger.info(
            f"Resumed from manaual huggingface checkpoint: {checkpoint_path}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model = unwrapped_model.from_pretrained(
            os.path.join(checkpoint_path, "model"))
        checkpoint_info = torch.load(
            os.path.join(checkpoint_path, "ckpt_info.pt"))
        resume_epoch = checkpoint_info["epoch"]
        sample_weights = checkpoint_info["sample_weights"]
        logger.info(f"train info restored: ckpt epoch {resume_epoch}")
        if os.path.exists(os.path.join(checkpoint_path, "ckpt_lr.pt")):
            checkpoint_lr = torch.load(
                os.path.join(checkpoint_path, "ckpt_lr.pt"))
            lr_scheduler.load_state_dict(checkpoint_lr["lr_scheduler"])
            logger.info(f"lr state restored")
        else:
            logger.info(f"Warning, no LR info in checkpoint dir!")
        if os.path.exists(os.path.join(checkpoint_path, "ckpt_opt.pt")):
            try:
                checkpoint_opt = torch.load(
                    os.path.join(checkpoint_path, "ckpt_opt.pt"))
                optimizer.load_state_dict(checkpoint_opt["optimizer"])
                logger.info(f"opt state restored")
            except:
                logger.info(
                    f"error loading opt state due to cuda oom results from large adamw opt... will resume without opt state")
        else:
            logger.info(f"Warning, no OPT info in checkpoint dir!")
    elif os.path.isdir(os.path.join(args.output_dir, ACCELERATOR_RESUME_CKPT_FOLDER)):
        checkpoint_path = os.path.join(
            args.output_dir, ACCELERATOR_RESUME_CKPT_FOLDER)
        accelerator.load_state(os.path.join(checkpoint_path, "ckpt_state"))
        logger.info(f"accelerator state recovered")

        checkpoint_info = torch.load(
            os.path.join(checkpoint_path, "ckpt_info.pt"))
        resume_epoch = checkpoint_info["epoch"]
        sample_weights = checkpoint_info["sample_weights"]
        logger.info(f"train info restored: ckpt epoch {resume_epoch}")

    elif os.path.isdir(os.path.join(args.output_dir, AUTO_RESUME_CKPT_FOLDER)):
        checkpoint_path = os.path.join(
            args.output_dir, AUTO_RESUME_CKPT_FOLDER)
        logger.info(
            f"Resumed from automatic resumable checkpoint: {checkpoint_path}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model = unwrapped_model.from_pretrained(
            os.path.join(checkpoint_path, "model"))
        checkpoint_info = torch.load(
            os.path.join(checkpoint_path, "ckpt_info.pt"))
        resume_epoch = checkpoint_info["epoch"]
        sample_weights = checkpoint_info["sample_weights"]
        logger.info(f"train info restored: ckpt epoch {resume_epoch}")

        try:
            checkpoint_opt = torch.load(
                os.path.join(checkpoint_path, "ckpt_opt.pt"))
            optimizer.load_state_dict(checkpoint_opt["optimizer"])
            logger.info(f"opt state restored")

            checkpoint_lr = torch.load(
                os.path.join(checkpoint_path, "ckpt_lr.pt"))
            lr_scheduler.load_state_dict(checkpoint_lr["lr_scheduler"])
            logger.info(f"lr state restored")
        except:
            logger.info(
                f"error loading opt/lr state, possible due to cuda oom due to large adamw opt... will resume without opt state")
    else:
        logger.info(
            f"Fresh new run from model checkpoint {args.model_name_or_path}")

    # ================ Recalculate steps etc
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # ================ Init Tracker
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        if args.report_to == "wandb":
            accelerator.init_trackers(
                project_name="lmkb-at-scale", config=experiment_config, init_kwargs={"wandb": {"notes": args.run_name}})
        else:
            accelerator.init_trackers("kllm", experiment_config)

    # ================ Training Entrance
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if resume_epoch is not None:
        starting_epoch = resume_epoch + 1
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch

    do_init_eval = True
    if do_init_eval:
        logger.info("***** Initial evaluation *****")
        em_score, f1_score, final_result = evaluate(
            model, valid_dataloader, accelerator, args, tokenizer, epoch="init", save_result=True)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    epoch = -1
    best_em, best_f1 = 0, 0
    best_em_epoch, best_f1_epoch = starting_epoch, starting_epoch
    for epoch in range(starting_epoch, args.num_train_epochs):
        timestamp = time.time()

        # ================ ImSmp
        # prerform importance sampling and re-prepare train datalaoder
        # reference: https://github.com/huggingface/accelerate/issues/752
        accelerator.wait_for_everyone()
        del train_dataloader
        with accelerator.main_process_first():
            print("epoch {} performing importance sampling (main process={})".format(
                epoch, accelerator.is_main_process))
            sample_size = max(1, int(len(train_dataset) * sample_ratio))
            if len(train_dataset) < LARGE_DATASET_SIZE:
                sample_indices = torch.Tensor(list(WeightedRandomSampler(
                    sample_weights, sample_size, replacement=False)))
                sample_indices = sample_indices.tolist()
            else:  # workround fro numpy bug: https://github.com/numpy/numpy/pull/6131
                sample_distro = np.asarray(sample_weights).astype("float32")
                sample_distro = sample_distro / \
                    np.sum(sample_distro)  # normalize
                sample_indices = np.random.choice(np.arange(
                    len(train_dataset)), size=sample_size, replace=False, p=sample_distro)
            sample_dataset = train_dataset.select(sample_indices)
            train_dataloader = DataLoader(
                sample_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(
                    tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=args.per_device_train_batch_size
            )
            train_dataloader = accelerator.prepare(train_dataloader)
            print("epoch {} train dataloader perpared (main process={})".format(
                epoch, accelerator.is_main_process))

        # ================ standard training
        model.train()
        total_loss = 0
        instance_losses = []
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss

                # compute instance loss
                accelerator.wait_for_everyone()
                logits = outputs.logits  # outputs[1]
                labels = batch["labels"]
                loss_fn = CrossEntropyLoss(reduction="none", ignore_index=-100)
                # (bs * labels.shape[-1])
                instance_loss = loss_fn(
                    logits.view(-1, logits.size(-1)), labels.view(-1))
                instance_loss = torch.sum(instance_loss.view(
                    logits.size(0), -1), dim=1)  # (bs)
                instance_losses += instance_loss.tolist()

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item(
                    ) / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    logger.info(
                        "unable to save per-step due to per-epoch sampling")
                    continue
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(
                                args.output_dir, output_dir)
                        save_with_accelerate_imsmp(
                            accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        # update sample weights using instance losses
        accelerator.wait_for_everyone()
        with accelerator.main_process_first():
            assert len(instance_losses) <= len(sample_indices), "{} instance losses more than {} sample indices".format(
                len(instance_losses), len(sample_indices))
            sample_weights[sample_indices[:len(instance_losses)]] = torch.Tensor(
                instance_losses).type(torch.float32)
            print("epoch {} finished. sample weight updated! (main proc={})".format(
                epoch, accelerator.is_main_process))

        logger.info("Epoch {}: train time={}".format(epoch, str(
            datetime.timedelta(seconds=time.time() - timestamp))))

        if args.checkpointing_steps == "epoch":
            save_with_accelerate_ckpt(
                accelerator, model, sample_weights, args, epoch)
            save_with_accelerate_full(accelerator, sample_weights, args, epoch)

        if epoch % EVAL_EPOCH == 0:
            em_score, f1_score, result_df = evaluate(
                model, valid_dataloader, accelerator, args, tokenizer, epoch=epoch, save_result=True)
            accelerator.log(
                {"eval/em": em_score, "eval/f1": f1_score, "eval/epoch": epoch})
            if em_score > best_em:
                best_em_epoch = epoch
                best_em = em_score
            if f1_score > best_f1:
                best_f1_epoch = epoch
                best_f1 = f1_score
            if EARLY_STOPPING:
                if best_em >= EARLY_STOPPING_SCORE:
                    break
                if best_f1 >= EARLY_STOPPING_SCORE:
                    break
                if epoch - best_em_epoch > EARLY_STOPPING_EPOCH:
                    break
                if epoch - best_f1_epoch > EARLY_STOPPING_EPOCH:
                    break

    # final evaluation step
    do_final_eval = False
    if epoch > 0 and do_final_eval:
        logger.info("***** Final evaluation *****")
        em_score, f1_score, final_result = evaluate(
            model, valid_dataloader, accelerator, args, tokenizer, epoch="final", save_result=True)

    if args.with_tracking:
        accelerator.end_training()

    do_final_save = False
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        if do_final_save:
            save_with_accelerate_ckpt(
                accelerator, model, sample_weights, args, "final")


if __name__ == "__main__":
    main()
