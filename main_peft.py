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


PAD_TOKEN="<|pad|>"


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
    'meta-llama/Llama-3.1-8B',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
)

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,
    use_rslora=True,
    bias='none',
    init_lora_weights="gaussian",
    task_type=TaskType.CAUSAL_LM,
)

dataset = load_dataset(
    "json",
    data_files={"train":"Data/MultiTask_Training_Data/train_data.json", "validation":"Data/MultiTask_Training_Data/validation_data.json"}
)

peft_model = get_peft_model(base_model, lora_config)

# 4: Train the PeftModel (same way as training base model)
sft_config = SFTConfig(
    output_dir="./Output",
    dataset_text_field="text",
    max_seq_length=2048,
    num_train_epochs=1, 
    gradient_accumulation_steps=4,
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1, 
    fp16=True, 
    dataset_kwargs={
        "add_special_tokens":False,
        "append_concat_token":False,
    },
)
trainer = SFTTrainer(
    peft_model,
    args=sft_config,
    # pert_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

peft_model.print_trainable_parameters()

# 5: Start Training
trainer.train()

trainer.save_model()




# def main():
#     # set_random_seed(42)
    
#     tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]
#     tokenizer.add_special_tokens({"pad_token":PAD_TOKEN})
#     tokenizer.padding_side='right'

#     response_template = "<|end_header_id|>"
#     collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  # Generation Part 제외한 Instruction, FewShot Example 부분은 -100으로 마스킹하여 파인튜닝 성능개선

    
#     base_model = AutoModelForCausalLM.from_pretrained(
#         'meta-llama/Llama-3.1-8B',
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         device_map='auto',
#     )

#     lora_config = LoraConfig(
#         lora_alpha=16,
#         lora_dropout=0.05,
#         r=32,
#         use_rslora=True,
#         bias='none',
#         init_lora_weights="gaussian",
#         task_type=TaskType.CAUSAL_LM,
#     )

#     dataset = load_dataset(
#         "json",
#         data_files={"train":"/Data/MultiTask_Training_Data/Dolphin_MultiTask_Shuffled_train.json", "validation":"Data/MultiTask_Training_Data/Dolphin_MultiTask_Shuffled_validation.json"}
#     )

#     peft_model = get_peft_model(base_model, lora_config)

#     # 4: Train the PeftModel (same way as training base model)
#     sft_config = SFTConfig(
#         output_dir="./LLaMA3_1/fine_tuning/fine_tuned_models",
#         dataset_text_field="Text_FewShot",
#         max_seq_length=3072,
#         num_train_epochs=1, 
#         gradient_accumulation_steps=4,
#         per_device_train_batch_size=1, 
#         per_device_eval_batch_size=1, 
#         fp16=True, 
#         dataset_kwargs={
#             "add_special_tokens":False,
#             "append_concat_token":False,
#         },
#     )
#     trainer = SFTTrainer(
#         peft_model,
#         args=sft_config,
#         # pert_config=lora_config,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["validation"],
#         tokenizer=tokenizer,
#         data_collator=collator,
#     )

#     peft_model.print_trainable_parameters()

#     # 5: Start Training
#     trainer.train()

#     trainer.save_model()
    

# if __name__ == "__main__":
#     main()
