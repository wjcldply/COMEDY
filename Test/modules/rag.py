from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import logging

from modules.utils.types import RetrieveMetadata

logger = logging.getLogger(__name__)
kiwi = Kiwi()

model = SentenceTransformer("all-MiniLM-L6-v2")

TOP_K = 3


def embed_sentence_transformer(text):
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def tokenize_kiwi(text):
    tokens = kiwi.tokenize(text)
    return [token.form for token in tokens]


def rag_bm25(test_case_dict, backbone, topk=TOP_K, **kwargs):
    turn_strings = []
    for session in test_case_dict["history_sessions"]:
        for turn_idx in range(0, len(session), 2):
            turn_string = f"User: {session[turn_idx]['utterance']}\nBot: {session[turn_idx+1]['utterance']}"
            turn_strings.append(turn_string)
    tokenized_turns = [tokenize_kiwi(turn_string) for turn_string in turn_strings]
    bm25 = BM25Okapi(tokenized_turns)
    current_turn = test_case_dict["current_session_test"][-1]
    current_turn_string = f"{current_turn['utterance']}"
    tokenized_current_turn = tokenize_kiwi(current_turn_string)
    bm25_scores = bm25.get_scores(tokenized_current_turn)
    # Pick the top k turns
    top_k_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:topk]
    top_k_turns = [turn_strings[i] for i in top_k_indices]

    current_session_string = ""
    for turn in test_case_dict[
        "current_session_test"
    ]:  # turn -> {"speaker":"user", "utterence":"..."}
        if turn["speaker"] == "user":
            current_session_string += f"User: {turn['utterance']}\n"
        else:
            current_session_string += f"Bot: {turn['utterance']}\n"

    system_message = """Look at the Previous relevant Dialogue History and Generate Personalized Response for the Current Session.
    
    Previous relavant Session's Dialogue History: """

    for turn in top_k_turns:
        system_message += turn
        system_message += "\n===\n"

    system_message += """

    Current Session: """
    system_message += current_session_string

    formatted_prompt = [{"role": "system", "content": system_message}]
    response = backbone(formatted_prompt=formatted_prompt)
    return response, RetrieveMetadata(retrieved_turns=top_k_turns)


def rag_faiss(test_case_dict, backbone, topk=TOP_K, **kwargs):
    turn_strings = []
    for session in test_case_dict["history_sessions"]:
        for turn_idx in range(0, len(session), 2):
            turn_string = f"User: {session[turn_idx]['utterance']}\nBot: {session[turn_idx+1]['utterance']}"
            turn_strings.append(turn_string)

    embeddings = np.array(
        [embed_sentence_transformer(turn_string) for turn_string in turn_strings]
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    current_turn = test_case_dict["current_session_test"][-1]
    current_turn_string = f"{current_turn['utterance']}"
    current_turn_embedding = np.array([embed_sentence_transformer(current_turn_string)])

    _, top_k_indices = index.search(current_turn_embedding, topk)

    top_k_turns = [turn_strings[i] for i in top_k_indices[0]]

    current_session_string = ""
    for turn in test_case_dict["current_session_test"]:
        if turn["speaker"] == "user":
            current_session_string += f"User: {turn['utterance']}\n"
        else:
            current_session_string += f"Bot: {turn['utterance']}\n"

    system_message = """Look at the Previous relevant Dialogue History and Generate Personalized Response for the Current Session.

    Previous relevant Session's Dialogue History: """

    for turn in top_k_turns:
        system_message += turn
        system_message += "\n===\n"

    system_message += """

    Current Session: """
    system_message += current_session_string

    formatted_prompt = [{"role": "system", "content": system_message}]
    response = backbone(formatted_prompt=formatted_prompt)
    return response, RetrieveMetadata(retrieved_turns=top_k_turns)
