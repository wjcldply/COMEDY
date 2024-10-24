from openai import OpenAI


def backbone_gpt(formatted_prompt: list, model: str):
    """
    formatted_prompt (list) : [{"role":"system", "content":"..."}]
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=formatted_prompt,
    )
    response_text = response.choices[0].message.content
    return response_text
