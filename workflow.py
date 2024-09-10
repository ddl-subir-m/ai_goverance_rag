from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional

from retrieval import retrieve
from answer_generation import generate_answer, check_relevance
from question_processing import rephrase_question
from constants import MAX_ITERATIONS, MAX_REPHRASE_ATTEMPTS

import nltk

class State(TypedDict):
    question: str
    vectorstore: Any
    docs: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    is_relevant: bool
    is_partially_relevant: bool
    context_relevant: bool
    relevance_info: Optional[str]
    rephrase_count: int
    iteration_count: int
    need_retrieval: bool

def ensure_nltk_packages():
    nltk_data_dir = './.venv/nltk_data'
    packages = ['punkt', 'averaged_perceptron_tagger']

    for package in packages:
        try:
            nltk.data.find(f'{package}', paths=[nltk_data_dir])
            print(f"{package} is already up to date.")
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)

def setup_workflow():
    # Ensure NLTK packages are present
    ensure_nltk_packages()

    workflow = StateGraph(State)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("rephrase_question", rephrase_question)
    
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        lambda x: (
            "end" if x["is_relevant"] or x["iteration_count"] >= MAX_ITERATIONS or x.get("rephrase_count", 0) > MAX_REPHRASE_ATTEMPTS
            else "rephrase" if x["is_partially_relevant"] or not x["context_relevant"]
            else "retrieve"
        ),
        {
            "retrieve": "retrieve",
            "rephrase": "rephrase_question",
            "end": END
        }
    )
    workflow.add_edge("rephrase_question", "retrieve")
    
    workflow.set_entry_point("retrieve")
    
    return workflow.compile()