import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from peft import PeftModel, PeftConfig


def load_llama(model_path: str, lora_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    terminators = [tokenizer.eos_token_id]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # left -> right 수정함

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    if lora_path != "Base":
        model = PeftModel.from_pretrained(model, lora_path)
    return tokenizer, model


def backbone_llama(formatted_prompt: list, model, tokenizer):
    """
    formatted_prompt (list) : [{"role":"system", "content":"..."}]
    """
    terminators = [tokenizer.eos_token_id]
    untokenized_prompt = tokenizer.apply_chat_template(
        formatted_prompt, tokenize=False, add_generation_prompt=True
    )
    tokenized_prompt = tokenizer(untokenized_prompt, return_tensors="pt").to(
        model.device
    )

    prompt_length = tokenized_prompt["input_ids"].shape[1]

    encoded_response = model.generate(
        tokenized_prompt["input_ids"],
        attention_mask=tokenized_prompt["attention_mask"],
        max_new_tokens=2048,
        eos_token_id=terminators[0],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )

    generated_tokens = encoded_response[0][prompt_length:]
    decoded_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded_response
