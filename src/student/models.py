from __future__ import annotations

import uuid
from typing import List, Union
from pydantic import BaseModel, Field

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class AnsweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    sources: List[MinimalSource]
    answer: str

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

class StudentSearchResults(BaseModel):
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(BaseModel):
    answer: str

class MinimalAnswerResults(BaseModel):
    search_results: List[MinimalAnswer]
    k: int

class MinimalSearchResultsList(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class Chunk(BaseModel):
    chunk_id: int
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    kind: str  # "code" | "text"


# Definición de RagDataset para corregir el error de importación
class RagDataset(BaseModel):
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalSearchResults]
    answers: List[MinimalAnswer]
