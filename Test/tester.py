import json
import logging
from pathlib import Path
import subprocess
import sys
from tqdm import tqdm

import torch
from functools import partial

from modules.utils.types import TaskType
from modules import memory, rag
from modules.context import context_window_prompting
from modules.llama import backbone_llama, load_llama
from modules.openai import backbone_gpt
from modules.prompt_composer import comedy
from modules.utils.logger import init_logger

RETRIEVE_TOP_K = 3
logger = init_logger(is_main=True)


def load_testset(test_file_path: str):
    test_file_path = Path(test_file_path)
    if test_file_path.is_file():  # Test Dataset 이미 있으면 스킵
        logger.info("✅ Test File Found")
    else:  # Test Dataset 없으면 생성
        logger.warning("❌ Test File NOT Found, Generating One...")
        subprocess.run(["python", "test_data_build.py"])
        logger.info("✅ Test File Successfully Created")

    # Load Dataset
    with open(test_file_path, "r") as infile:
        data = json.load(infile)
    test_cases = data["session4"]
    for case in test_cases:
        case["results"] = dict()
    return test_cases  # [{'history_sessions', 'current_session_original', 'current_session_test', 'id', 'results'}, ... x1000]


def test(test_cases: list, test_configs: list):
    for config in tqdm(test_configs, desc="Iterating Test Configurations"):
        if config["model"] == "gpt-4o-mini":
            backbone = partial(backbone_gpt, model=config["model"])
            keyword = f"{config['type']} | {config['model']}"
        else:
            tokenizer, model = load_llama(config["model"], config["lora_path"])
            # backbone_llama(model=model, tokenizer=tokenizer)
            backbone = partial(backbone_llama, model=model, tokenizer=tokenizer)
            keyword = f"{config['type']} | {config['model']} | {config['lora_path']}"

        for case in tqdm(
            test_cases, desc="Iterating Test Cases"
        ):  # test_cases: [{'history_sessions', 'current_session_original', 'current_session_test', 'id'}, ... x1000]
            if config["type"] == TaskType.COMEDY:
                # result = comedy(case, config['model'], config['lora_path'])
                result = comedy(case, backbone)
            elif config["type"] == TaskType.CONTEXT_WINDOW_PROMPTING:
                # result = context_window_prompting(case, config['model'], config['lora_path'])
                result = context_window_prompting(case, backbone)
            elif config["type"] == TaskType.RAG_BM25:
                # result = rag(case, config['model'], config['lora_path'])
                result = rag.rag_bm25(case, backbone, topk=RETRIEVE_TOP_K)
            elif config["type"] == TaskType.RAG_FAISS:
                # result = rag(case, config['model'], config['lora_path'])
                result = rag.rag_faiss(case, backbone, topk=RETRIEVE_TOP_K)
            elif config["type"] == TaskType.MEMORY:
                # result = memory(case, config['model'], config['lora_path'])
                result = memory(case, backbone)
            else:
                raise Exception

            # Assuming 'Result' variable holds string values only
            case["results"][keyword] = result
        if config["model"] != "gpt-4o-mini":
            del model
            del tokenizer
            torch.cuda.empty_cache()

    logger.info("✅ Test Complete | 🔥 Saving...")
    with open("test_results.json", "w") as outfile:
        json.dump(test_cases, outfile, indent=4, ensure_ascii=False)
    logger.info("✅ Done")


test_configs = [
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-1B-LoRA32/final/'},
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'Base'},
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-3B-LoRA32/final/'},
    # {'type':'COMEDY', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'Base'},
    ## {'type':'COMEDY', 'model':"gpt-4o-mini"},
    ## {'type':'COMEDY', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'../Models/llama3.1-8B-LoRA32/final/'},
    ## {'type':'COMEDY', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'Base'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-1B-LoRA32/final/'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-1B-Instruct', 'lora_path':'Base'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'COMEDY/Models/llama3.2-3B-LoRA32/final/'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.2-3B-Instruct', 'lora_path':'Base'},
    # {'type':'Context Window Prompting', 'model':"gpt-4o-mini"},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'../Models/llama3.1-8B-LoRA32/final/'},
    # {'type':'Context Window Prompting', 'model':'meta-llama/Llama-3.1-8B-Instruct', 'lora_path':'Base'},
    {"type": TaskType.RAG_BM25, "model": "gpt-4o-mini"},
    {"type": TaskType.RAG_FAISS, "model": "gpt-4o-mini"},
    # {'type':'RAG', 'model':, 'lora_pathlora_path':},
    # {'type':'MEMORY', 'model':, 'lora_pathlora_path':},
]

if __name__ == "__main__":

    test_cases = load_testset("Test/test_data.json")
    test(test_cases[:1], test_configs)
