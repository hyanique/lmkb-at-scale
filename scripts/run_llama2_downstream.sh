export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

memo="LLaMA-2 downstream examples on a specific checkpoint"

checkpoint_name="some_checkpoint_name"
checkpoint_path="some_checkpoint_file.pt"

# ==============================================================
# Fixed-Form Information Recall
# ==============================================================
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
NUM_EPOCH=0
MIX_PREC='bf16'
MODEL_NAME="meta-llama/Llama-2-7b-hf"
dt=`date '+%Y%m%d_%H%M%S'`
run_name=llama_v2_7b-Valid10K_${checkpoint_name}
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    llama2_ft.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --resume_from_checkpoint ${checkpoint_path} \
    --use_slow_tokenizer \
    --finetune_dataset "Valid10K" \
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
    --report_to tensorboard \
    --logging_steps 500 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
&>> ${log}


run_name=llama_v2_7b-MemRel_${checkpoint_name}
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    llama2_ft.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --resume_from_checkpoint ${checkpoint_path} \
    --use_slow_tokenizer \
    --finetune_dataset "MemRel" \
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
    --output_dir runs/${run_name}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 500 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
&>> ${log}

# ==============================================================
# Free-Form Information Recall
# ==============================================================
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
NUM_EPOCH=15
MIX_PREC='bf16'
MODEL_NAME="meta-llama/Llama-2-7b-hf"
dt=`date '+%Y%m%d_%H%M%S'`
run_name=llama_v2_7b-PQA_${checkpoint_name}
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    kllama14ft2.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --resume_from_checkpoint ${checkpoint_path} \
    --use_slow_tokenizer \
    --finetune_dataset "PopQA" \
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
    --output_dir runs/${run_name}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 500 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
&>> ${log}

# ==============================================================
# General Missing Fact Completion
# ==============================================================
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
NUM_EPOCH=0
MIX_PREC='bf16'
MODEL_NAME="meta-llama/Llama-2-7b-hf"
dt=`date '+%Y%m%d_%H%M%S'`
run_name=llama_v2_7b-KBC_${checkpoint_name}
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    llama2_ft.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --resume_from_checkpoint ${checkpoint_path} \
    --use_slow_tokenizer \
    --finetune_dataset "KBC" \
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
    --output_dir runs/${run_name}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 500 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
&>> ${log}

fname=runs/${run_name}/init-eval_results.csv
python kbc-eval.py --file_name $fname >> ${log}

# ==============================================================
# Inverse Reasoning
# ==============================================================
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
NUM_EPOCH=0
MIX_PREC='bf16'
MODEL_NAME="meta-llama/Llama-2-7b-hf"
dt=`date '+%Y%m%d_%H%M%S'`
run_name=llama_v2_7b-OneHop-Nill_${checkpoint_name}
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    llama2_ft.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --resume_from_checkpoint ${checkpoint_path} \
    --use_slow_tokenizer \
    --finetune_dataset "OneHop" \
    --reasoning_format "nill" \
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
    --output_dir runs/${run_name}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 500 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
&>> ${log}


# ==============================================================
# Compositional Reasoning
# ==============================================================
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
NUM_EPOCH=0
MIX_PREC='bf16'
MODEL_NAME="meta-llama/Llama-2-7b-hf"
dt=`date '+%Y%m%d_%H%M%S'`
run_name=llama_v2_7b-TwoHop-Nill_${checkpoint_name}
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    llama2_ft.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --resume_from_checkpoint ${checkpoint_path} \
    --use_slow_tokenizer \
    --finetune_dataset "TwoHop" \
    --reasoning_format "nill" \
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
    --report_to tensorboard \
    --logging_steps 500 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
&>> ${log}

dt=`date '+%Y%m%d_%H%M%S'`
run_name=llama_v2_7b-TwoHop-Verify_${checkpoint_name}
log=logs/${dt}-${run_name}.log.txt
accelerate launch \
    --mixed_precision $MIX_PREC \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file fineune_stage3_no_offloading_accelerate.conf \
    llama2_ft.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --resume_from_checkpoint ${checkpoint_path} \
    --use_slow_tokenizer \
    --finetune_dataset "TwoHop" \
    --reasoning_format "verify" \
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
    --output_dir runs/${run_name}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 500 \
    --seed 42 \
    --run_name $run_name\
    --logger_name ${dt}-${run_name}.logger.txt \
&>> ${log}