#!/bin/bash
memo="T5-large downstream examples on a specific checkpoint"

ckpt_path="some_checkpoint_file.pt"
ckpt_name="some_checkpoint_name"

# ==============================================================
# Fixed-Form Information Recall
# ==============================================================
knowledge_fpath="data_dir/entity-tail.json"
knowledge_spec="EntTail"
mode="eval"
model="t5-large"
seed=6
nworkers=8
mprecision="bf16"
auto_wb="False"
dataset="wikidata"
ebsize=512
comp_metric_interval=1
srclen=40
tgtlen=40
genmax=40
manual_checkpoint=${ckpt_path}
dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Large-MemEval-${ckpt_name}-${knowledge_spec}
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
CUDA_VISIBLE_DEVICES=0,1 python t5km_bf16.py \
    --mode ${mode} \
    --model ${model} \
    --seed ${seed} \
    --nworkers ${nworkers} \
    --run_name ${run_name} \
    --out_dir ${save_dir} \
    --dataset ${dataset} \
    --mprecision ${mprecision} \
    --knowledge_fpath ${knowledge_fpath} \
    --manual_checkpoint ${manual_checkpoint} \
    --ebsize ${ebsize} \
    --comp_metric_interval ${comp_metric_interval} \
    --last_validation full \
    --srclen ${srclen} \
    --tgtlen ${tgtlen} \
    --genmax ${genmax} \
    --auto_wb ${auto_wb} \
>> ${log}

# ==============================================================
# Free-Form Information Recall
# ==============================================================
mode="train"
model="t5-large"
seed=6
nworkers=8
mprecision="bf16"
auto_wb="False"
dataset="popqa-split"
lr="1e-3"
opt="adafactor"
scheduler="constant_lr"
nepoch=30
bsize=256
ebsize=512
use_importance_sampling="False"
comp_metric_interval=1
srclen=40
tgtlen=40
genmax=40
dataset="popqa-split"
dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Large-PQA-Finetune-${ckpt_name}
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
manual_start_epoch=0
CUDA_VISIBLE_DEVICES=0,1 python t5km_bf16.py \
    --mode ${mode} \
    --model ${model} \
    --seed ${seed} \
    --nworkers ${nworkers} \
    --run_name ${run_name} \
    --out_dir ${save_dir} \
    --mprecision ${mprecision} \
    --manual_checkpoint ${ckpt_path} \
    --manual_start_epoch ${manual_start_epoch} \
    --dataset ${dataset} \
    --lr ${lr} \
    --opt ${opt} \
    --scheduler ${scheduler} \
    --nepoch ${nepoch} \
    --bsize ${bsize} \
    --ebsize ${ebsize} \
    --use_importance_sampling ${use_importance_sampling} \
    --comp_metric_interval ${comp_metric_interval} \
    --srclen ${srclen} \
    --tgtlen ${tgtlen} \
    --genmax ${genmax} \
    --auto_wb ${auto_wb} \
>> ${log}



mode="eval"
model="t5-large"
seed=6
nworkers=8
mprecision="bf16"
auto_wb="False"
dataset="popqa-triplet"
ebsize=512
use_importance_sampling="False"
comp_metric_interval=1
srclen=40
tgtlen=40
genmax=40
dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Large-PQA-Triplet-${ckpt_name}
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
manual_start_epoch=0
date >> ${log}
echo ${run_name} >> ${log}
echo ${memo} >> ${log}
echo >> ${log}
CUDA_VISIBLE_DEVICES=0,1 python t5km_bf16.py \
    --mode ${mode} \
    --model ${model} \
    --seed ${seed} \
    --nworkers ${nworkers} \
    --run_name ${run_name} \
    --out_dir ${save_dir} \
    --mprecision ${mprecision} \
    --manual_checkpoint ${ckpt_path} \
    --manual_start_epoch ${manual_start_epoch} \
    --dataset ${dataset} \
    --ebsize ${ebsize} \
    --use_importance_sampling ${use_importance_sampling} \
    --comp_metric_interval ${comp_metric_interval} \
    --srclen ${srclen} \
    --tgtlen ${tgtlen} \
    --genmax ${genmax} \
    --auto_wb ${auto_wb} \
>> ${log}

# ==============================================================
# General Missing Fact Completion
# ==============================================================
mode="eval"
model="t5-large"
seed=6
nworkers=8
mprecision="bf16"
auto_wb="False"
ebsize=512
comp_metric_interval=1
srclen=40
tgtlen=40
genmax=40
dataset="missing-facts"
knowledge_fpath="data_dir/annotated_missing_fact.json"
manual_checkpoint=${ckpt_path}
dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Large-KBC-${ckpt_name}
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
CUDA_VISIBLE_DEVICES=0,1 python t5km_bf16.py \
    --mode ${mode} \
    --model ${model} \
    --seed ${seed} \
    --nworkers ${nworkers} \
    --run_name ${run_name} \
    --out_dir ${save_dir} \
    --dataset ${dataset} \
    --mprecision ${mprecision} \
    --manual_checkpoint ${manual_checkpoint} \
    --knowledge_fpath ${knowledge_fpath} \
    --ebsize ${ebsize} \
    --comp_metric_interval ${comp_metric_interval} \
    --last_validation full \
    --srclen ${srclen} \
    --tgtlen ${tgtlen} \
    --genmax ${genmax} \
    --auto_wb ${auto_wb} \
>> ${log}
fname=${save_dir}/eval00_results.csv
python kbc_eval.py --file_name $fname &>> ${log}

# ==============================================================
# Inverse Reasoning
# ==============================================================
mode="eval"
model="t5-large"
seed=6
nworkers=8
mprecision="bf16"
auto_wb="False"
ebsize=512
comp_metric_interval=1
srclen=40
tgtlen=40
genmax=40
dataset="onehopSROBackward"
manual_checkpoint=${ckpt_path}
dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Large-${dataset}-${ckpt_name}
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
CUDA_VISIBLE_DEVICES=0,1 python t5km_bf16.py \
    --mode ${mode} \
    --model ${model} \
    --seed ${seed} \
    --nworkers ${nworkers} \
    --run_name ${run_name} \
    --out_dir ${save_dir} \
    --mprecision ${mprecision} \
    --dataset ${dataset} \
    --manual_checkpoint ${manual_checkpoint} \
    --ebsize ${ebsize} \
    --comp_metric_interval ${comp_metric_interval} \
    --last_validation full \
    --srclen ${srclen} \
    --tgtlen ${tgtlen} \
    --genmax ${genmax} \
    --auto_wb ${auto_wb} \
>> ${log}

# ==============================================================
# Compositional Reasoning
# ==============================================================
mode="eval"
model="t5-large"
seed=6
nworkers=8
mprecision="bf16"
auto_wb="False"
ebsize=512
comp_metric_interval=1
srclen=40
tgtlen=40
genmax=40
dataset="twohopSROb2c"
manual_checkpoint=${ckpt_path}
dt=`date '+%Y%m%d_%H%M%S'`
run_name=T5Large-${dataset}-${ckpt_name}
log=logs/${dt}-${run_name}.log.txt
save_dir=runs/${run_name}
mkdir -p ${save_dir}
CUDA_VISIBLE_DEVICES=0,1 python t5km_bf16.py \
    --mode ${mode} \
    --model ${model} \
    --seed ${seed} \
    --nworkers ${nworkers} \
    --run_name ${run_name} \
    --out_dir ${save_dir} \
    --dataset ${dataset} \
    --mprecision ${mprecision} \
    --manual_checkpoint ${manual_checkpoint} \
    --ebsize ${ebsize} \
    --comp_metric_interval ${comp_metric_interval} \
    --last_validation full \
    --srclen ${srclen} \
    --tgtlen ${tgtlen} \
    --genmax ${genmax} \
    --auto_wb ${auto_wb} \
>> ${log}