import os
import json
from tqdm import tqdm
from pprint import pprint
from functools import partial
from modules.comedy import comedy_task3
from modules.openai import backbone_gpt
from modules.llama import backbone_llama, load_llama

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score
from collections import Counter

def calculate_bleu(reference, candidate, n):
    weights = tuple((1.0 / n if i < n else 0.0) for i in range(4))  # BLEU-n weights
    return sentence_bleu([reference.split()], candidate.split(), weights=weights, smoothing_function=SmoothingFunction().method1)

def calculate_f1(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    ref_count = Counter(ref_tokens)
    cand_count = Counter(cand_tokens)
    
    # Calculate the number of true positives, false positives, and false negatives
    tp = sum((ref_count & cand_count).values())
    fp = sum((cand_count - ref_count).values())
    fn = sum((ref_count - cand_count).values())
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1


def create_dummy_EvalData(validation_data):
    with open(validation_data, 'r') as file:
        validationset = json.load(file)
    for item in validationset["session4"]:
        item["gold"] = {
            "task1_response":"this is just random place holder", 
            "task2_response":"this is also just random place holder"
        }
    # Save the updated data back to the file
    with open(validation_data, 'w') as file:
        json.dump(validationset, file, indent=4, ensure_ascii=False,)


def merge_results(inferenced_data, test_data, validation_data):
    '''
    inferenced_data (jsonl) file directory : 테스트 파일로 라마 모델에서 생성 진행한 결과파일
    test_data (json) file directory : 테스트 수행용 데이터 파일 ("id"로 매치해서 잘라둔 "current_session_test" 가져와야함)
    validation_data (json) file directory  : "id"로 매치해서 상응하는 "personaInfo", "gold" 값 가져와야 함
    '''
    # 파일 로드
    # all_test_cases = []
    with open(inferenced_data, 'r') as file1:
        all_test_cases = json.load(file1)

    assert len(all_test_cases)==200

    with open(test_data, 'r') as file2:
        testset = json.load(file2)  # {"session2", "session3", "session4"}
    with open(validation_data, 'r') as file3:
        validationset = json.load(file3)
    
    for testcase in all_test_cases:  # testcase : {"id","type", "model", "lora_path", "final_response", "metadata":{"task1_response","task2_response","current_sessions"}}
        target_id = testcase["id"]  # 기존 데이터셋들(테스트셋, 밸리데이션v2) 순회

        dict_containing_current_session_test = next((item for item in testset["session4"] if item.get("id") == target_id), None)
        current_session_test = dict_containing_current_session_test["current_session_test"]  # 가져온 Current Session 전체 내용 (모델 답변 생성 직전까지)
        history_sessions = dict_containing_current_session_test["history_sessions"]
        dict_containing_personaInfo = next((item for item in validationset["session4"] if item.get("id") == target_id), None)
        # pprint(dict_containing_personaInfo)
        personaInfo = dict_containing_personaInfo["metadata"]["personaInfo"]  # 가져온 personaInfo
        gold_task_123 = dict_containing_personaInfo["gold"]  # 가져온 Task1, 2, 3에 대한 GOLD 결과물 {"task3_response":"string", "task2_response":"string", "task1_response":["string", "string", "string"],}
        # task3_str = ""
        # for elem in gold_task_123["task3_response"]:
        #     task3_str+=elem
        # gold_task_123["task3_response"] = task3_str

        testcase["metadata"]["current_session_test"] = current_session_test
        testcase["metadata"]["personaInfo"] = personaInfo
        testcase["metadata"]["history_sessions"] = history_sessions
        testcase["gold"] = gold_task_123

    # 길이 체크
    assert len(all_test_cases)==200

    # 파일 저장
    with open(inferenced_data.replace('.jsonl', '')+'_merged.jsonl', 'w') as file:
        json.dump(
            all_test_cases,
            file,
            indent=4,
            ensure_ascii=False,
        )
        
    return all_test_cases


def re_inference_openai(merged_results_file):
    '''
    merged_results (jsonl) file directory  : need to be loaded into list -> [{"id","type", "model", "lora_path", "final_response", "metadata":{"task1_response","task2_response","current_sessions", "current_session_test", "personaInfo", "history_sessions"}}] (len:200)

    GPT-4o-mini 인퍼런스 재진행하여 "final_response_gpt4omini"에 저장
    '''
    # 파일 로드
    # all_test_cases = []
    with open(merged_results_file, 'r') as file1:
        all_test_cases = json.load(file1)
    assert len(all_test_cases)==200

    # 재 인퍼런스 후 데이터에 추가
    backbone = partial(backbone_gpt, model="gpt-4o-mini")
    for testcase in tqdm(all_test_cases):
        testcase["final_response_gpt4omini"] = comedy_task3(testcase, backbone)

    # 길이 체크
    assert len(all_test_cases)==200

    # 파일 저장
    with open(merged_results_file, 'w') as file:
        json.dump(
            all_test_cases,
            file,
            indent=4,
            ensure_ascii=False,
        )
        
    return all_test_cases


def re_inference_llama(merged_results_file, model_path='meta-llama/Llama-3.1-8B-Instruct', lora_path='Base'):
    '''
    merged_results (jsonl) file directory  : need to be loaded into list -> [{"id","type", "model", "lora_path", "final_response", "metadata":{"task1_response","task2_response","current_sessions", "current_session_test", "personaInfo", "history_sessions"}}] (len:200)

    Llama-3.1-8B-Base 인퍼런스 재진행하여 "final_response_llamaBase"에 저장
    '''
    # 파일 로드
   # all_test_cases = []
    with open(merged_results_file, 'r') as file1:
        all_test_cases = json.load(file1)
    assert len(all_test_cases)==200

    # 재 인퍼런스 후 데이터에 추가
    tokenizer, model = load_llama(model_path, lora_path='Base')
    backbone = partial(backbone_llama, model=model, tokenizer=tokenizer)
    for testcase in tqdm(all_test_cases):
        testcase["final_response_llamaBase"] = comedy_task3(testcase, backbone)

    # 길이 체크
    assert len(all_test_cases)==200

    # 파일 저장
    with open(merged_results_file, 'w') as file:
        json.dump(
            all_test_cases,
            file,
            indent=4,
            ensure_ascii=False,
        )
        
    return all_test_cases

def calc_scores_vs_gold(merged_results_file):
    '''
    merged_results (jsonl) file directory  : need to be loaded into list -> [{"id","type", "model", "lora_path", "final_response", "final_response_gpt4omini", "final_response_llamaBase","metadata":{"task1_response","task2_response","current_sessions", "current_session_test", "personaInfo", "history_sessions"}}] (len:200)
    '''
    # 파일 로드
    # all_test_cases = []
    with open(merged_results_file, 'r') as file1:
        all_test_cases = json.load(file1)
    assert len(all_test_cases)==200

    for testcase in tqdm(all_test_cases):
        # ["metadata"]["task1_response"] vs ["gold"]["task1_response"] -> BLEU 1, 2 | F1 Scores 계산 -> ["metadata"]["task1_scores"]["bleu1"], ["metadata"]["task1_scores"]["bleu2"], ["metadata"]["task1_scores"]["f1"] 저장
        generated_task1_response_list = testcase["metadata"]["task1_response"]
        generated_task1_response = ""
        for elem in generated_task1_response_list:
            generated_task1_response += elem
        gold_task1_response_list = testcase["gold"]["task1_response"]
        gold_task1_response = ""
        for elem in gold_task1_response_list:
            gold_task1_response += elem
        

        testcase["metadata"]["task1_scores"] = {}
        testcase["metadata"]["task1_scores"]["bleu1"] = calculate_bleu(gold_task1_response, generated_task1_response, 1)
        testcase["metadata"]["task1_scores"]["bleu2"] = calculate_bleu(gold_task1_response, generated_task1_response, 2)
        testcase["metadata"]["task1_scores"]["f1"] = calculate_f1(gold_task1_response, generated_task1_response)
        
        # ["metadata"]["task2_response"] vs ["gold"]["task2_response"] -> BLEU 1, 2 | F1 Scores 계산 -> ["metadata"]["task2_scores"]["bleu1"], ["metadata"]["task2_scores"]["bleu2"], ["metadata"]["task2_scores"]["f1"] 저장
        generated_task2_response = testcase["metadata"]["task2_response"]
        gold_task2_response = testcase["gold"]["task2_response"]
        testcase["metadata"]["task2_scores"] = {}
        testcase["metadata"]["task2_scores"]["bleu1"] = calculate_bleu(gold_task2_response, generated_task2_response, 1)
        testcase["metadata"]["task2_scores"]["bleu2"] = calculate_bleu(gold_task2_response, generated_task2_response, 2)
        testcase["metadata"]["task2_scores"]["f1"] = calculate_f1(gold_task2_response, generated_task2_response)

    # 길이 체크
    assert len(all_test_cases)==200

    # 파일 저장
    with open(merged_results_file, 'w') as file:
        json.dump(
            all_test_cases,
            file,
            indent=4,
            ensure_ascii=False,
        )
        
    return all_test_cases


inference_results = [
    'results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_base_10_25.jsonl',
    'results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_tuned_10_24.jsonl',
    'results/COMEDY_meta-llama_Llama-3.2-3B-Instruct_tuned_10_24.jsonl',
    'results/COMEDY_meta-llama_Llama-3.2-3B-Instruct_base_10_24.jsonl',
]

inference_results_merged = [
    'results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_base_10_25_merged.jsonl',
    'results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_tuned_10_24_merged.jsonl',
    'results/COMEDY_meta-llama_Llama-3.2-3B-Instruct_tuned_10_24_merged.jsonl',
    'results/COMEDY_meta-llama_Llama-3.2-3B-Instruct_base_10_24_merged.jsonl',
]

for inference_data_dir, inference_data_merged_dif in zip(inference_results, inference_results_merged):
    print(f'working on {inference_data_dir}')
    print('🔥 Merging...')
    merge_results(
        inferenced_data=inference_data_dir,
        test_data='test_data.json',
        validation_data='validation_data_v3.json',
    )
    print('✅ Merged !')

    # print('🔥 Re-Inferencing on LLaMA3.1 8B...')
    # re_inference_llama(inference_data_merged_dif)
    # print('✅ Done !')

    print('🔥 Re-Inferencing on Open AI API...')
    re_inference_openai(inference_data_merged_dif)
    print('✅ Done !')

    print('🔥 Calculating BLEU/F1 Scores...')
    calc_scores_vs_gold(inference_data_merged_dif)
    print('✅ Done !')