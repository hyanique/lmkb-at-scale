#!/bin/bash
memo="T5-large training example"

dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Large-Training-BF16
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
mode="train"
model="t5-large"
seed=6
nworkers=8
auto_wb="True"
dataset="wikidata"
knowledge_fpath="some_knowledge_dataset.json"
data_sample_ratio=1
lr="1e-3"
opt="adafactor"
scheduler="constant_lr"
mprecision="bf16"
nepoch=50
bsize=300
ebsize=512
use_importance_sampling="True"
importance_sampling_ratio="0.3"
nvalid=10000
comp_metric_interval=1
last_validation="split"
srclen=40
tgtlen=40
genmax=40
CUDA_VISIBLE_DEVICES=2,3 python t5km_bf16.py \
    --mode ${mode} \
    --model ${model} \
    --seed ${seed} \
    --nworkers ${nworkers} \
    --run_name ${run_name} \
    --out_dir ${save_dir} \
    --dataset ${dataset} \
    --knowledge_fpath ${knowledge_fpath} \
    --data_sample_ratio ${data_sample_ratio} \
    --lr ${lr} \
    --opt ${opt} \
    --scheduler ${scheduler} \
    --mprecision ${mprecision} \
    --nepoch ${nepoch} \
    --bsize ${bsize} \
    --ebsize ${ebsize} \
    --use_importance_sampling ${use_importance_sampling} \
    --importance_sampling_ratio ${importance_sampling_ratio} \
    --nvalid ${nvalid} \
    --comp_metric_interval ${comp_metric_interval} \
    --last_validation ${last_validation} \
    --srclen ${srclen} \
    --tgtlen ${tgtlen} \
    --genmax ${genmax} \
    --auto_wb ${auto_wb} \
>> ${log}