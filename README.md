# Language Models over Large-Scale Knowledge Base

## Introduction

This repository provides code and datasets for paper “Language Models over Large-Scale Knowledge Base: on Capacity, Flexibility and Reasoning for New Facts”. 

## Dependencies

- python=3.9.18
- torch==2.0.0
- accelerate==0.24.1
- numpy==1.24.1
- deepspeed==0.12.2

We also use a public fork of HuggingFace transformer library to address the left padding problem that may impact the inference results: `pip install git+https://github.com/yizhongw/transformers.git@left_padding`

## Data

You can download our world knowledge dataset from [link](https://drive.google.com/file/d/10dgqyklm1D7MUBH8x1J5TGcs0GEaTvGM/view?usp=sharing). Alternative, you can compile  your own knowledge dataset into a json file with each line formatted as 

```json
{"subj":"subject_text","rel":"relation_text","obj":"object_text"}
```

Other downstream dataset are available in the `/data` folder. 

## Code

Among five python files, `t5km.py` and `t5km_bf16.py` are for training and evaluating T5 models in FT32 and BF16 respectively. `llama2_km.py` is for training LLaMA-2 models and `llama2_ft.py` is for downstream finetuning and evaluating. In addition, `kbc-eval.py` is for computing the exact match and F1 scores for general missing fact completion, where multiple answer candidates are available. Several example bash scripts are included in the `/scripts` folder.