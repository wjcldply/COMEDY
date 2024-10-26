import json
from pprint import pprint

ids = ["6629822d-715b-4e56-ada8-42e1203a5093",
"8e9cf691-4832-4f2e-a1d4-8771ba907207",
"2c411952-13da-4543-af0c-24cab79773ca",
"9d13be20-18b9-432b-bdd9-6ac1a189222c",
"407aa811-f492-47bc-a395-f23cbccfcd71",
"efe9f951-f112-4407-ab6c-2c15d93ef8a7",
"c441748f-ddae-4426-991d-0249918be63a",
"01e6ad0b-0341-4a78-a0f7-d3b2bb7299da",
"8b47ea61-38d5-4b52-ac22-f3fd3ad0505b",
"003022e0-c10b-44a4-9b7b-2273549b6d7a",
"660d48db-7a41-4ef6-9107-e9a171705ada",
"30d5fd12-a8bc-4bf5-97fa-315b8a92739b",
"e41d8e32-1c32-4a04-b86d-6bc5a20a310f",
"fe4425e1-bf03-4f13-85f1-ef2be440db10",
"91cd97c1-6ca4-4507-b3e4-0032afb0a8f2",
"ce534a13-639f-4204-8dfa-391c25ec4640",
"d38747bf-c985-4915-a703-0b32ba8c5eca",
"d8e22f6b-ded6-44d0-be1f-efab4d9b7d28",
"c14922db-90fe-49df-853b-22b60de73d34",
"308730de-66e5-4fc2-9656-c8dfe974232d",
"81b3e57a-cc6b-4059-aee0-2fc3f8e3e23a",
"22308a7a-9ad4-4d07-a0ad-4d202ab65de2",
"d63cdc92-e18e-4418-89c1-8c0bef283979",
"05816a43-2ffa-4124-920d-ee7b3e885e77",
"a92cc89e-1b70-4081-b58d-e9e7ce9838e7",
"b9d37038-3147-440a-a7b1-9372568d32e0",
"96c1a856-585a-4d6a-9a45-4e2069da0d21",
"01fac538-06fe-4435-aef4-6a35719f7363",
"d3a63fa6-74c5-462a-bf4d-ba1c1a3c44b9",
"5b750c21-b3e3-4fa3-a51e-72505e436464",]

print(len(ids))

cases = {
    "file_1_COMEDY_LLAMA_8B_GPT":"results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_tuned_10_24_merged.jsonl",
    "file_2_COMEDY_GPT":"results/COMEDY_gpt-4o-mini_base_10_24.jsonl",
    "file_3_RAG_BM25_GPT":"results/RAG_BM25_gpt-4o-mini_base_10_24.jsonl",
    "file_4_RAG_FAISS_GPT":"results/RAG_FAISS_gpt-4o-mini_base_10_24.jsonl",
    "file_5_CONTEXT_FULL_GPT":"results/CONTEXT_WINDOW_PROMPTING_gpt-4o-mini_base_10_24.jsonl",
    "file_6_CONTEXT_LESS_GPT":"results/CONTEXT_FREE_gpt-4o-mini_base_10_24.jsonl"
}

with open("results/COMEDY_meta-llama_Llama-3.1-8B-Instruct_tuned_10_24_merged.jsonl", 'r') as file1:
    f1 = json.load(file1)
with open("results/COMEDY_gpt-4o-mini_base_10_24.jsonl", 'r') as file1:
    f2 = json.load(file1)
with open("results/RAG_BM25_gpt-4o-mini_base_10_24.jsonl", 'r') as file1:
    f3 = json.load(file1)
with open("results/RAG_FAISS_gpt-4o-mini_base_10_24.jsonl", 'r') as file1:
    f4 = json.load(file1)
with open("results/CONTEXT_WINDOW_PROMPTING_gpt-4o-mini_base_10_24.jsonl", 'r') as file1:
    f5 = json.load(file1)
with open("results/CONTEXT_FREE_gpt-4o-mini_base_10_24.jsonl", 'r') as file1:
    f6 = json.load(file1)


keys_from_merged = ["current_session_test", "history_sessions", "final_response_gpt4omini","final_response_llamaBase"]
keys_from_rest = "final_response"


new_jsons = []
for id in ids:
    f1_item = next((item for item in f1 if item.get("id") == id), None)
    # pprint(f1_item)
    f2_item = next((item for item in f2 if item.get("id") == id), None)
    f3_item = next((item for item in f3 if item.get("id") == id), None)
    f4_item = next((item for item in f4 if item.get("id") == id), None)
    f5_item = next((item for item in f5 if item.get("id") == id), None)
    f6_item = next((item for item in f6 if item.get("id") == id), None)

    all_combinations_retrieved = {
        "History":f1_item["metadata"]["history_sessions"],
        "Current":f1_item["metadata"]["current_session_test"],
        "Session Memory":f1_item["metadata"]["task1_response"],
        "Compressed Memory":f1_item["metadata"]["task2_response"],
        "COMEDY_LLAMA(GPT)":f1_item["final_response_gpt4omini"],
        "COMEDY_LLAMA(LLAMA)":f1_item["final_response_llamaBase"],
        "COMEDY_GPT(GPT)":f2_item["final_response"],
        "RAG_BM25(GPT)":f3_item["final_response"],
        "RAG_FAISS(GPT)":f4_item["final_response"],
        "CONTEXT_FULL(GPT)":f5_item["final_response"],
        "CONTEXT_LESS(GPT)":f6_item["final_response"],
    }
    new_jsons.append(all_combinations_retrieved)

pprint(new_jsons)
print(len(new_jsons))

with open('human_eval_file.json', 'w') as f:
    for item in new_jsons:
        f.write(json.dumps(item, indent=4, ensure_ascii=False) + "\n")