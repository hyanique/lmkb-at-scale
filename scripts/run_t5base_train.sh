#!/bin/bash
memo="T5-base training example"

dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Base-Training
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
mode="train"
model="t5-base"
seed=6
nworkers=8
auto_wb="True"
dataset="wikidata"
knowledge_fpath="some_knowledge_dataset.json"
data_sample_ratio=1
lr="1e-3"
opt="adafactor"
scheduler="constant_lr"
nepoch=50
bsize=512
ebsize=1024
use_importance_sampling="True"
importance_sampling_ratio="0.3"
nvalid=10000
comp_metric_interval=1
last_validation="split"
srclen=40
tgtlen=40
genmax=40
python t5km.py \
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