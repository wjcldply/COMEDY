import json
from pathlib import Path
import subprocess
from pprint import pprint
from tqdm import tqdm
from openai import OpenAI
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch
from functools import partial

def load_testset():
    test_file_name = 'test_data.json'
    test_file_path = Path(test_file_name)
    if test_file_path.is_file():  # Test Dataset 이미 있으면 스킵
        print('✅ Test File Found')
    else:  # Test Dataset 없으면 생성
        print('❌ Test File NOT Found, Generating One...')
        subprocess.run(['python', 'test_data_build.py'])
        print('✅ Test File Successfully Created')

    # Load Dataset
    with open(test_file_name, 'r') as infile:
        data = json.load(infile)
    test_cases = data["session4"]
    for case in test_cases:
        case["results"] = dict()
    return test_cases  # [{'history_sessions', 'current_session_original', 'current_session_test', 'id', 'results'}, ... x1000]

def backbone_gpt(formatted_prompt:list, model:str):
    '''
    formatted_prompt (list) : [{"role":"system", "content":"..."}]
    '''
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=formatted_prompt,
    )
    response_text = response.choices[0].message.content
    return response_text

def load_llama(model_path:str, lora_path:str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    terminators = [tokenizer.eos_token_id]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # left -> right 수정함
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
    if lora_path != 'Base':
        model = PeftModel.from_pretrained(model, lora_path)
    return tokenizer, model

def backbone_llama(formatted_prompt:list, model, tokenizer):
    '''
    formatted_prompt (list) : [{"role":"system", "content":"..."}]
    '''
    terminators = [tokenizer.eos_token_id]
    untokenized_prompt = tokenizer.apply_chat_template(formatted_prompt, tokenize=False, add_generation_prompt=True)
    tokenized_prompt = tokenizer(untokenized_prompt, return_tensors='pt').to(model.device)
    encoded_response = model.generate(
        tokenized_prompt['input_ids'],
        attention_mask=tokenized_prompt['attention_mask'],
        max_new_tokens = 2048,
        eos_token_id=terminators[0],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )
    decoded_response = tokenizer.decode(encoded_response[0], skip_special_tokens=True)

    return decoded_response


# def comedy(test_case_dict, model_path, lora_path):
def comedy(test_case_dict, backbone):
    '''
    input: 
        test_case_dict (dict): {'history_sessions', 'current_session_test'}
        model_path (str): model_path (hf transformers path || open ai api model name)
        lora_path (str, None): lora adapter directory path (Defaults to None, Ignored when model_path is OPEN AI API Model Name)
    output: 
        (dict) Generated (Personalized) Output
    '''
    
    ########## Task 1 : Session-Level Memory Summarization ##########
    task1_system_message_begin = """This is a memory description generation task.
    In this task, based on the dialogue between two individuals, you will create objective memory descriptions for both individuals. 
    The descriptions should be formatted as [xxx|xxx|xxx], where each 'xxx' represents a distinct memory.
    Each memory should use the speaker's name as the subject, and all relevant dialogue content must be captured. 
    Make sure to refer to the participants as [USER] and [ASSISTANT].
    Separate different memories with '|'.
    
    Final Output Format:
    [XXX|XXX|XXX]
    
    The dialogue content is:
    """
    task1_system_message_end = "The output is:"

    task1_formatted_prompts = []
    for session in test_case_dict["history_sessions"]:  # session -> [{"speaker":"user", "utterence":"..."}, {"speaker":"bot", "utterence":"..."}, {}, {}, ...]
        session_string = ""
        for turn in session:  # turn -> {"speaker":"user", "utterence":"..."}
            if turn["speaker"] == "user":
                session_string += f"User: {turn['utterance']}\n"
            else:
                session_string += f"Bot: {turn['utterance']}\n"
        task1_formatted_prompts.append([{"role":"system", "content":task1_system_message_begin+session_string+task1_system_message_end}])
    
    task1_reponses = []  # -> [[...|...|...], [...|...|...], [...|...|...]] 3개 세션에 대한 세션 단위 요약들이 각각 저장됨, 각각 string 타입임
    for task1_formatted_prompt in task1_formatted_prompts:
        response = backbone(formatted_prompt=task1_formatted_prompt)
        task1_reponses.append(response)
    
    ########## Task 2 : Memory Compression ##########
    task2_system_message_begin = """This is a task about customizing user descriptions, relationship descriptions, and event descriptions.
    The text output is divided into three parts:
    The first part is the user description, mainly including a summary of the user's information.
    The second part describes the relationship between the user and the robot.
    The third part describes the events shared by the user and the robot.
    Based on the reference materials, extract and summarize different information such as the user's personality traits and behavior patterns.
    It is important to record and include all information about the user from various aspects in the user description, without any omissions, resulting in an objective user description.
    The second part is the relationship description between the user and the robot, describing the level of intimacy shown in the dialogue.
    The third part is the description of events shared by the user and the robot, summarizing events that have occurred in the dialogue.
    In the output description, list specific examples mentioned in the reference materials as much as possible, retaining some interesting information.
    Use [USER] and [ASSISTANT] to refer to participants instead of specific names like "User" or "Bot".

    Final Output Format:
    User Description: Your Answer.
    Relationship Description: Your Answer.
    Event Description: Your Answer.

    The reference material is """
    
    task2_system_message_end = """.
    The output is:"""

    task2_concatenated_summaries = ""
    for task1_response in task1_reponses:
        task2_concatenated_summaries += task1_response
    task2_formatted_prompt = [{"role":"system", "content":task2_system_message_begin+task2_concatenated_summaries+task2_system_message_end}]
    task2_response = backbone(formatted_prompt=task2_formatted_prompt)
    task2_compressed_memory = task2_response

    ########## Task 3 : Memory-based Generatioin ##########
    current_session_string = ""
    for turn in test_case_dict["current_session_test"]:  # turn -> {"speaker":"user", "utterence":"..."}
        if turn["speaker"] == "user":
            current_session_string += f"User: {turn['utterance']}\n"
        else:
            current_session_string += f"Bot: {turn['utterance']}\n"

    system_message_3_begin = """[Human] This is a memory-based dialogue generation task.
    Given a dialogue and related memory content, please generate a response that is consistent with the memory content and reasonable within the context of the dialogue. 
    You just need to output the answer without any prefix like Bot:.

    Dialogue: """ + current_session_string
    
    system_message_3_end = """
    
    Memory: """ + task2_compressed_memory

    task3_formatted_prompt = [{"role":"system", "content":system_message_3_begin+system_message_3_end}]
    task3_response = backbone(formatted_prompt=task3_formatted_prompt)
    
    return task3_response
    

# def context_window_prompting(test_case_dict, model_path, lora_path):
def context_window_prompting(test_case_dict, backbone):
    '''
    input: 
        test_case_dict (dict): {'history_sessions', 'current_session_test'}
        model_path (str): model_path (hf transformers path || open ai api model name)
        lora_path (str, None): lora adapter directory path (Defaults to None, Ignored when model_path is OPEN AI API Model Name)
    output: 
        (str) Generated (Personalized) Output
    '''

    history_sessions_string = ""
    for session in test_case_dict['history_sessions']:  # session -> [{"speaker":"user", "utterence":"..."}, {"speaker":"bot", "utterence":"..."}, {}, {}, ...]
        session_string = ""
        for turn in session:  # turn -> {"speaker":"user", "utterence":"..."}
            if turn['speaker'] == 'user':
                session_string += f"User: {turn['utterance']}\n"
            else:
                session_string += f"Bot: {turn['utterance']}\n"
        history_sessions_string += session_string
        
    current_session_string = ""
    for turn in test_case_dict['current_session_test']:  # turn -> {"speaker":"user", "utterence":"..."}
        if turn['speaker'] == 'user':
            current_session_string += f"User: {turn['utterance']}\n"
        else:
            current_session_string += f"Bot: {turn['utterance']}\n"

    # All Session into System Prompt
    system_message = """Look at the Previous Session's Dialogue History and Generate Personalized Response for the Current Session.
    
    Previous Session's Dialogue History: """ 
    system_message += history_sessions_string
    system_message += """

    Current Session: """
    system_message += current_session_string

    formatted_prmopt = [{"role":"system", "content":system_message}]
    response = backbone(formatted_prmopt=formatted_prmopt)

    return response


def rag(test_case_dict, model_path, lora_path):
    '''
    To-Be-Implemented
    '''

def memory(test_case_dict, model_path, lora_path):
    '''
    To-Be-Implemented
    '''


def test(test_cases:list, test_configs:list):
    for config in tqdm(test_configs, desc='Iterating Test Configurations'):
        if config['model'] == 'gpt-4o-mini':
            backbone = partial(backbone_gpt, model=config['model'])
            keyword = f"{config['type']} | {config['model']}"
        else:
            tokenizer, model = load_llama(config['model'], config['lora_path'])
            # backbone_llama(model=model, tokenizer=tokenizer)
            backbone = partial(backbone_llama, model=model, tokenizer=tokenizer)
            keyword = f"{config['type']} | {config['model']} | {config['lora_path']}"
        for case in tqdm(test_cases, desc='Iterating Test Cases'):  # test_cases: [{'history_sessions', 'current_session_original', 'current_session_test', 'id'}, ... x1000]
            if config['type'] == 'COMEDY':
                # result = comedy(case, config['model'], config['lora_path'])
                result = comedy(case, backbone)
            elif config['type'] == 'Context Window Prompting':
                # result = context_window_prompting(case, config['model'], config['lora_path'])
                result = context_window_prompting(case, backbone)
            elif config['type'] == 'RAG':
                # result = rag(case, config['model'], config['lora_path'])
                result = rag(case, backbone)
            elif config['type'] == 'MEMORY':
                # result = memory(case, config['model'], config['lora_path'])
                result = memory(case, backbone)
            else:
                raise Exception

            # Assuming 'Result' variable holds string values only
            case['results'][keyword] = result
        if config['model'] != 'gpt-4o-mini':
            del model
            del tokenizer
            torch.cuda.empty_cache()

    print('✅ Test Complete | 🔥 Saving...')
    with open('test_results.json', 'w') as outfile:
        json.dump(test_cases, outfile, indent=4)
    print('✅ Done')


test_cases = load_testset()
test_configs = [
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-1B-LoRA32/final/'},
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'Base'},
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-3B-LoRA32/final/'},
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'Base'},
    {'type':'COMEDY', 'model':"gpt-4o-mini"},
    {'type':'COMEDY', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'../Models/llama3.1-8B-LoRA32/final/'},
    {'type':'COMEDY', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'Base'},

    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-1B-LoRA32/final/'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'Base'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-3B-LoRA32/final/'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'Base'},
    {'type':'Context Window Prompting', 'model':"gpt-4o-mini"},
    {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'../Models/llama3.1-8B-LoRA32/final/'},
    {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'Base'},

    # {'type':'RAG', 'model':, 'lora_pathlora_path':},
    
    # {'type':'MEMORY', 'model':, 'lora_pathlora_path':},
]


test(test_cases[:5], test_configs)