from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class TaskType(str, Enum):
    COMEDY = "COMEDY"
    CONTEXT_WINDOW_PROMPTING = "CONTEXT_WINDOW_PROMPTING"
    RAG_BM25 = "RAG_BM25"
    RAG_FAISS = "RAG_FAISS"
    MEMORY = "MEMORY"
    CONTEXT_FREE = "CONTEXT_FREE"


@dataclass
class ComedyMetadata:
    task1_response: str
    task2_response: str
    current_sessions: List[str]


@dataclass
class ContextMetadata:
    history_sessions: str
    current_sessions: List[str]


@dataclass
class RetrieveMetadata:
    retrieved_turns: list
    current_sessions: List[str]


@dataclass
class TaskResult:
    id: int
    type: TaskType
    model: Optional[str]
    lora_path: Optional[str]
    final_response: str
    metadata: Optional[Union[ComedyMetadata, ContextMetadata, RetrieveMetadata]] = None
