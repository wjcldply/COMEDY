# def context_window_prompting(test_case_dict, model_path, lora_path):
def context_window_prompting(test_case_dict, backbone):
    """
    input:
        test_case_dict (dict): {'history_sessions', 'current_session_test'}
        model_path (str): model_path (hf transformers path || open ai api model name)
        lora_path (str, None): lora adapter directory path (Defaults to None, Ignored when model_path is OPEN AI API Model Name)
    output:
        (str) Generated (Personalized) Output
    """

    history_sessions_string = ""
    for session in test_case_dict[
        "history_sessions"
    ]:  # session -> [{"speaker":"user", "utterence":"..."}, {"speaker":"bot", "utterence":"..."}, {}, {}, ...]
        session_string = ""
        for turn in session:  # turn -> {"speaker":"user", "utterence":"..."}
            if turn["speaker"] == "user":
                session_string += f"User: {turn['utterance']}\n"
            else:
                session_string += f"Bot: {turn['utterance']}\n"
        history_sessions_string += session_string

    current_session_string = ""
    for turn in test_case_dict[
        "current_session_test"
    ]:  # turn -> {"speaker":"user", "utterence":"..."}
        if turn["speaker"] == "user":
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

    formatted_prompt = [{"role": "system", "content": system_message}]
    response = backbone(formatted_prompt=formatted_prompt)

    return response
