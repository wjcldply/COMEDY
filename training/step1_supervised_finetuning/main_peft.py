#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils import tensorboard

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    TaskType,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import PartialState, prepare_pippy

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import set_random_seed

os.environ["WANDB_PROJECT"]="KO-COMEDY"

PAD_TOKEN="<|pad|>"

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--context_window', type=int, default=4096, help='Context Window Size for LLaMA3.1 (defaults to 4096)')
parser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA')
parser.add_argument('--epochs', type=int, default=1, help='# of Epochs')
parser.add_argument('--per_device_batch_size', type=int, default=1, help='# of Batches per GPU')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='# of examples to accumulate gradients before taking a step')
parser.add_argument('--checkpointing_ratio', type=float, default=0.25, help='Percentage of Epochs to be Completed Before a Model Saving Happens')
parser.add_argument('--fp16',action='store_true', help='whether or not to use FP16')
parser.add_argument('--wandb_run_name', type=str, default='base', help='Wandb Logging Name for this Training Run')
parser.add_argument('--test',action='store_true', help='use very small dataset(~100) to validate the fine-tuning process')
args = parser.parse_args()

# set_random_seed(42)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
tokenizer.add_special_tokens({"pad_token":PAD_TOKEN})
tokenizer.padding_side='right'

response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  # Generation Part 제외한 Instruction, FewShot Example 부분은 -100으로 마스킹하여 파인튜닝 성능개선


base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
)

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=args.lora_rank,
    use_rslora=True,
    bias='none',
    init_lora_weights="gaussian",
    task_type=TaskType.CAUSAL_LM,
)

if args.test:
    train_dir = "Data/ko_COMEDY/MultiTask_Train_SMALL.json"
    validation_dir = "Data/ko_COMEDY/MultiTask_Validation_SMALL.json"
else:
    train_dir = "Data/ko_COMEDY/MultiTask_Train.json"
    validation_dir = "Data/ko_COMEDY/MultiTask_Validation.json"

dataset = load_dataset(
    "json",
    data_files={"train":train_dir, "validation":validation_dir}
)

peft_model = get_peft_model(base_model, lora_config)

# 4: Train the PeftModel (same way as training base model)
sft_config = SFTConfig(
    output_dir="./Models/",
    dataset_text_field="text",
    max_seq_length=args.context_window,
    num_train_epochs=args.epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_train_batch_size=args.per_device_batch_size, 
    per_device_eval_batch_size=args.per_device_batch_size, 
    fp16=args.fp16,
    dataset_kwargs={
        "add_special_tokens":False,
        "append_concat_token":False,
    },
    save_strategy="steps",
    save_steps=args.checkpointing_ratio,
    evaluation_strategy="steps",
    eval_steps=args.checkpointing_ratio,
    report_to="wandb",
    run_name=args.wandb_run_name,
    logging_steps=1,
)
trainer = SFTTrainer(
    peft_model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

peft_model.print_trainable_parameters()

# 5: Start Training
trainer.train()

trainer.save_model()
