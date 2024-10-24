# def comedy(test_case_dict, model_path, lora_path):
from modules.utils.types import ComedyMetadata


def comedy(test_case_dict, backbone):
    """
    input:
        test_case_dict (dict): {'history_sessions', 'current_session_test'}
        model_path (str): model_path (hf transformers path || open ai api model name)
        lora_path (str, None): lora adapter directory path (Defaults to None, Ignored when model_path is OPEN AI API Model Name)
    output:
        (dict) Generated (Personalized) Output
    """

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
    for session in test_case_dict[
        "history_sessions"
    ]:  # session -> [{"speaker":"user", "utterence":"..."}, {"speaker":"bot", "utterence":"..."}, {}, {}, ...]
        session_string = ""
        for turn in session:  # turn -> {"speaker":"user", "utterence":"..."}
            if turn["speaker"] == "user":
                session_string += f"User: {turn['utterance']}\n"
            else:
                session_string += f"Bot: {turn['utterance']}\n"
        task1_formatted_prompts.append(
            [
                {
                    "role": "system",
                    "content": task1_system_message_begin
                    + session_string
                    + task1_system_message_end,
                }
            ]
        )

    task1_reponses = (
        []
    )  # -> [[...|...|...], [...|...|...], [...|...|...]] 3개 세션에 대한 세션 단위 요약들이 각각 저장됨, 각각 string 타입임
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
    task2_formatted_prompt = [
        {
            "role": "system",
            "content": task2_system_message_begin
            + task2_concatenated_summaries
            + task2_system_message_end,
        }
    ]
    task2_response = backbone(formatted_prompt=task2_formatted_prompt)
    task2_compressed_memory = task2_response

    ########## Task 3 : Memory-based Generatioin ##########
    current_session_string = ""
    for turn in test_case_dict[
        "current_session_test"
    ]:  # turn -> {"speaker":"user", "utterence":"..."}
        if turn["speaker"] == "user":
            current_session_string += f"User: {turn['utterance']}\n"
        else:
            current_session_string += f"Bot: {turn['utterance']}\n"

    system_message_3_begin = (
        """[Human] This is a memory-based dialogue generation task.
    Given a dialogue and related memory content, please generate a response that is consistent with the memory content and reasonable within the context of the dialogue. 
    You just need to output the answer without any prefix like Bot:.

    Dialogue: """
        + current_session_string
    )

    system_message_3_end = (
        """
    
    Memory: """
        + task2_compressed_memory
    )

    task3_formatted_prompt = [
        {"role": "system", "content": system_message_3_begin + system_message_3_end}
    ]
    task3_response = backbone(formatted_prompt=task3_formatted_prompt)

    return task3_response, ComedyMetadata(task1_response=task1_reponses, task2_response=task2_response)
