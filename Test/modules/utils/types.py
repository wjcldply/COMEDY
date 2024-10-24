from enum import Enum


class TaskType(str, Enum):
    COMEDY = "COMEDY"
    CONTEXT_WINDOW_PROMPTING = "CONTEXT_WINDOW_PROMPTING"
    RAG_BM25 = "RAG_BM25"
    RAG_FAISS = "RAG_FAISS"
    MEMORY = "MEMORY"
