import json

# Open the input text file and output text file
with open('./MultiTask_Training_Data/Dolphin_MultiTask_Shuffled_train.json', 'r') as infile, open('./MultiTask_Training_Data/comedy_zh_train_data.json', 'w') as outfile:
    # Read and process each line
    for line in infile:
        # Replace '[HUMAN]' with 'user'
        modified_line = line.replace('[Human]',  '<|start_header_id|>user<|end_header_id|>').replace('[AI]','<|start_header_id|>assistant<|end_header_id|>').replace('<s>','<|begin_of_text|>').replace('</s>','<|eot_id|>').replace('<\/s>', '<|eot_id|>')
        # Write the modified line to the output file
        outfile.write(modified_line)

print("Replacement complete and saved to output_file.txt")


# Open the input text file and output text file
with open('./MultiTask_Training_Data/Dolphin_MultiTask_Shuffled_validation.json', 'r') as infile, open('./MultiTask_Training_Data/comedy_zh_validation_data.json', 'w') as outfile:
    # Read and process each line
    for line in infile:
        # Replace '[HUMAN]' with 'user'
        modified_line = line.replace('[Human]', '<|start_header_id|>user<|end_header_id|>').replace('[AI]','<|start_header_id|>assistant<|end_header_id|>').replace('<s>','<|begin_of_text|>').replace('</s>','<|eot_id|>').replace('<\/s>', '<|eot_id|>')
        # Write the modified line to the output file
        outfile.write(modified_line)

print("Replacement complete and saved to output_file.txt")
