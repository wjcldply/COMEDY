from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class TaskType(str, Enum):
    COMEDY = "COMEDY"
    CONTEXT_WINDOW_PROMPTING = "CONTEXT_WINDOW_PROMPTING"
    RAG_BM25 = "RAG_BM25"
    RAG_FAISS = "RAG_FAISS"
    MEMORY = "MEMORY"

@dataclass
class ComedyMetadata:
    task1_response: str
    task2_response: str

@dataclass
class ContextMetadata:
    history_sessions: str

@dataclass
class RetrieveMetadata:
    retrieved_turns: list

@dataclass
class TaskResult:
    id: int
    type: TaskType
    model: Optional[str]
    lora_path: Optional[str]
    final_response: str
    metadata: Optional[Union[ComedyMetadata, ContextMetadata, RetrieveMetadata]] = None
