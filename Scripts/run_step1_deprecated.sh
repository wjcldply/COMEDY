#!/bin/bash

# DeepSpeed Team
CURRENT_TIME=$(TZ=UTC-8 date +"%Y-%m-%d-%H.%M.%S")

# ZERO_STAGE="--zero_stage 2"
ZERO_STAGE="--zero_stage 3"  # configures the DeepSpeed zero optimization stage for memory efficiency during training

MODEL_PATH=$1  # path to the model to be fine-tuned
OUTPUT=$2  # base directory where output data will be saved
LOG_PATH=$3  # directory where logs will be saved
TRN_FN=$4  # training data file path
DEV_FN=$5  # development/validation data file path

echo "MODEL_PATH: '${MODEL_PATH}'"
echo "OUTPUT: '${OUTPUT}'"
echo "LOG_PATH: '${LOG_PATH}'"
echo "TRN_FN: '${TRN_FN}'"
echo "DEV_FN: '${DEV_FN}'"

export TOKENIZERS_PARALLELISM=False

NUM_GPUS=$(nvidia-smi -L | wc -l)
GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))

TOTAL_SIZE=`wc -l ${TRN_FN}`  #  number of lines (samples) in the training file
echo "number of samples in trainset: ${TOTAL_SIZE}"

mkdir -p $OUTPUT/$CURRENT_TIME

# deepspeed --include localhost:$GPU_LIST \  # Use All GPUs
# --master_port 12390 \  # Defines the port for the master process
deepspeed --num_gpus=${NUM_GPUS} training/step1_supervised_finetuning/main.py \
   --model_name_or_path ${MODEL_PATH} \
   --train_data_path ${TRN_FN} \
   --valid_data_path ${DEV_FN} \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --lora_dim 16 \
   --fp16 \
   --data_output_path $OUTPUT/data \
   --max_seq_len 2048 \
   --learning_rate 1e-5  \
   --weight_decay 0.1 \
   --num_train_epochs 3 \
   --num_train_samples ${TOTAL_SIZE} \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 400 \
   --seed 42 \
   ${ZERO_STAGE} \
   --save_interval 2000 \
   --log_interval 100 \
   --eval_interval 1000 \
   --output_dir $OUTPUT/$CURRENT_TIME \
   --gradient_checkpointing \
   --tensorboard_path $LOG_PATH \
   &>$OUTPUT/train_$CURRENT_TIME.log&