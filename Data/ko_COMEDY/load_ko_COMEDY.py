from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
import pandas as pd
import json
from tqdm import tqdm
import os
import warnings; warnings.filterwarnings("ignore", category=SyntaxWarning)

# Load the dataset from the Hugging Face Hub
dataset_1 = load_from_disk('./task1_dataset')
dataset_2 = load_from_disk('./task2_dataset')
dataset_3 = load_from_disk('./task1_dataset')

# 3개 Task 데이터셋 Merge
dataset_merged = concatenate_datasets([dataset_1, dataset_2, dataset_3])

# Shuffling 및 Train-Test Split
train_test_split = dataset_merged.shuffle(seed=42).train_test_split(test_size=0.05)
dataset_dict = DatasetDict({
    "train":train_test_split['train'],
    "validation":train_test_split['test']
})
print(dataset_dict)

# Split된 데이터셋 저장
dataset_dict['train'].to_json('./MultiTask_Train_temp.json')
dataset_dict['validation'].to_json('./MultiTask_Validation_temp.json')
print('saved; temp files')

# 각 Split 대상으로 LLaMA3 Token 스타일 적용
with open('./MultiTask_Train_temp.json', 'r') as infile, open('./MultiTask_Train.json', 'w') as outfile:
    for line in infile:
        modified_line = line.replace('[Human]',  '<|start_header_id|>user<|end_header_id|>').replace('[AI]','<|start_header_id|>assistant<|end_header_id|>').replace('<s>','<|begin_of_text|>').replace('</s>','<|eot_id|>').replace('<\/s>', '<|eot_id|>')
        outfile.write(modified_line)
    print('saved; test dataset')

with open('./MultiTask_Validation_temp.json', 'r') as infile, open('./MultiTask_Validation.json', 'w') as outfile:
    for line in infile:
        modified_line = line.replace('[Human]',  '<|start_header_id|>user<|end_header_id|>').replace('[AI]','<|start_header_id|>assistant<|end_header_id|>').replace('<s>','<|begin_of_text|>').replace('</s>','<|eot_id|>').replace('<\/s>', '<|eot_id|>')
        outfile.write(modified_line)
    print('saved; validation dataset')

# 임시 저장된 Split 데이터셋 삭제
os.remove('./MultiTask_Train_temp.json')
os.remove('./MultiTask_Validation_temp.json')
print('removed; temp files')

# 저장된 데이터셋 로드 및 출력하여 검증
with open('./MultiTask_Train.json', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        print(entry)
        break
