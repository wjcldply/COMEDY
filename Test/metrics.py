import json
from pprint import pprint

inference_results_merged = [
    'results/COMEDY_meta-llama_Llama-3.2-3B-Instruct_base_10_24_merged.jsonl',
    'results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_base_10_25_merged.jsonl',
    'results/COMEDY_meta-llama_Llama-3.2-3B-Instruct_tuned_10_24_merged.jsonl',
    'results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_tuned_10_24_merged.jsonl',
]

# Dictionary to store scores for each file
all_scores = {}

for file_dir in inference_results_merged:
    with open(file_dir, 'r') as file:
        results = json.load(file)
    assert len(results) == 200
    
    task_1_scores = {"bleu1": 0, "bleu2": 0, "f1": 0}
    task_2_scores = {"bleu1": 0, "bleu2": 0, "f1": 0}
    
    for sample in results:
        task_1_scores["bleu1"] += sample["metadata"]["task1_scores"]["bleu1"]
        task_1_scores["bleu2"] += sample["metadata"]["task1_scores"]["bleu2"]
        task_1_scores["f1"] += sample["metadata"]["task1_scores"]["f1"]

        task_2_scores["bleu1"] += sample["metadata"]["task2_scores"]["bleu1"]
        task_2_scores["bleu2"] += sample["metadata"]["task2_scores"]["bleu2"]
        task_2_scores["f1"] += sample["metadata"]["task2_scores"]["f1"]
    
    # Compute averages
    task_1_scores = {k: v / 200 for k, v in task_1_scores.items()}
    task_2_scores = {k: v / 200 for k, v in task_2_scores.items()}

    # Print scores for debugging
    print(f'Avg. Scores for {file_dir} are:')
    pprint(task_1_scores)
    pprint(task_2_scores)
    
    # Store results in all_scores dictionary
    all_scores[file_dir] = {
        "task1_scores": task_1_scores,
        "task2_scores": task_2_scores
    }

# Write all scores to scores.json
with open('scores.json', 'w') as outfile:
    json.dump(all_scores, outfile, indent=4)