#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=96
EVAL_BATCH_SIZE=96
TOTAL_BATCH_SIZE=768
LEARNING_RATE=1e-5 
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
MAX_SEQ_LEN=64 
MAX_NEW_TOKEN=16
NUM_WORKER=32
NUM_EPOCH=20
MIX_PREC='bf16'
MODEL_NAME="meta-llama/Llama-2-7b-hf"
dt=`date '+%Y%m%d_%H%M%S'`
run_name=llama2_train
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    llama2_km.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --use_slow_tokenizer \
    --train_file some_knowledge_dataset.json  \
    --max_seq_length $MAX_SEQ_LEN \
    --max_new_token $MAX_NEW_TOKEN \
    --preprocessing_num_workers $NUM_WORKER \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --evaluation_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type constant \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $NUM_EPOCH \
    --output_dir output_dir/${run_name}/ \
    --with_tracking \
    --report_to tensorflow \
    --logging_steps 5000 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
    --checkpointing_steps epoch \
&>> ${log}