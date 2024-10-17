from datasets import load_dataset, DatasetDict

# Load the dataset from the Hugging Face Hub
dataset_1 = load_dataset('Nuo97/Dolphin_Task1')
dataset_2 = load_dataset('Nuo97/Dolphin_Task2')
dataset_3 = load_dataset('Nuo97/Dolphin_Task3')
dataset_dpo = load_dataset("json", data_files='./DPO_Training_Data/memory_dpo_train.json')

# Inspect the dataset
print('Before Split')
print(dataset_1)
print(dataset_2)
print(dataset_3)
print(dataset_dpo)

# Split Data
train_val_split_1 = dataset_1["train"].train_test_split(test_size=0.2)
train_val_split_2 = dataset_2["train"].train_test_split(test_size=0.2)
train_val_split_3 = dataset_3["train"].train_test_split(test_size=0.2)

# Split DatasetDict Creation
split_dataset_1 = DatasetDict({
    'train': train_val_split_1['train'],
    'validation': train_val_split_1['test']
})
split_dataset_2 = DatasetDict({
    'train': train_val_split_2['train'],
    'validation': train_val_split_2['test']
})
split_dataset_3 = DatasetDict({
    'train': train_val_split_3['train'],
    'validation': train_val_split_3['test']
})

# Inspect the dataset
print('After Split')
print(split_dataset_1)
print(split_dataset_2)
print(split_dataset_3)
print(dataset_dpo)

split_dataset_1.save_to_disk('./MultiTask_Training_Data/Dolphin_Task1')
split_dataset_2.save_to_disk('./MultiTask_Training_Data/Dolphin_Task2')
split_dataset_3.save_to_disk('./MultiTask_Training_Data/Dolphin_Task3')