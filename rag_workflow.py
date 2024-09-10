from constants import *
from dotenv import load_dotenv
from typing import List, Dict, TypedDict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, END
from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

import os
import torch

load_dotenv()
# Initialize ColBERT model
colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Initialize the sentence transformer model
sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)


class CodeBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def ingest_documents(root_folder: str):
    print(f"Loading documents from: {root_folder}")
    
    loader = DirectoryLoader(root_folder, recursive=True)
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")
    
    if not documents:
        print("No documents were loaded. Check the root_folder path and file contents.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Number of document splits: {len(splits)}")
    
    # Create CodeBERT embedding function
    codebert_embeddings = CodeBERTEmbeddings()
    
    # Create the vector store with CodeBERT embeddings
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=codebert_embeddings,
        collection_name="project_rag_collection",
        # persist_directory="./chroma_db"
    )

    print(f"Ingested {vectorstore._collection.count()} documents into Chroma")
    
    return vectorstore

def load_or_create_store(root_folder: str):
    print("Creating new store...")
    vectorstore = ingest_documents(root_folder)
    return vectorstore

# Function to retrieve documents
def retrieve(state):
    if not state["need_retrieval"]:
        print("Skipping retrieval as question hasn't changed.")
        return {"docs": state["docs"], "question": state["question"]}

    # Get the total number of documents in the vector store
    total_docs = state["vectorstore"]._collection.count()
    
    # Use the minimum of RETRIEVAL_TOP_K and total_docs
    k = min(RETRIEVAL_TOP_K, total_docs)
    

    # Perform similarity search with scores
    docs_and_scores = state["vectorstore"].similarity_search_with_relevance_scores(
        state["question"], 
        k=k
    )

    # Filter documents based on relevance score
    filtered_docs = [
        doc for doc, score in docs_and_scores 
        if score >= RELEVANCE_THRESHOLD
    ]

    print(f"Number of documents retrieved: {len(filtered_docs)}")
    
    if not filtered_docs:
        print("No documents retrieved. Returning empty list.")
        return {"docs": [], "question": state["question"]}
    
    # Use the minimum of RERANK_TOP_K and the number of retrieved docs
    rerank_k = min(RERANK_TOP_K, len(filtered_docs))
    
    reranked_docs = colbert_model.rerank(
        state["question"], 
        [doc.page_content for doc in filtered_docs], 
        k=rerank_k
    )
    
    # Convert reranked_docs to the expected format
    formatted_docs = [
        {"content": doc["content"], "metadata": doc["metadata"] if "metadata" in doc else {}} 
        for doc in reranked_docs
    ]
    
    return {"docs": formatted_docs, "question": state["question"]}

# Function to generate an answer
def generate_answer(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the question based on the given context."),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ])
    
    chain = prompt | ChatOpenAI(model=GENERATE_ANSWER_MODEL) | StrOutputParser()
    
    answer = chain.invoke({
        "context": "\n\n".join([doc["content"] for doc in state["docs"]]),
        "question": state["question"]
    })
    
    return {"answer": answer}

# Function to check answer relevance
def check_relevance(state):
    context_relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXT_RELEVANCE_SYSTEM_PROMPT),
        ("human", "Context: {context}\n\nQuestion: {question}\n\nIs this context relevant to the question?"),
    ])
    
    answer_relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_RELEVANCE_SYSTEM_PROMPT),
        ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nIs this answer relevant and supported by the context?"),
    ])
    
    chain = ChatOpenAI(model=CHECK_RELEVANCE_MODEL) | StrOutputParser()
    
    context = "\n\n".join([doc["content"] for doc in state["docs"]]) if state["docs"] else ""
    
    if not context.strip():
        print("No context available. Marking as not relevant.")
        return {
            "answer": state["answer"],
            "docs": state["docs"],
            "is_relevant": False,
            "is_partially_relevant": False,
            "relevance_info": "No context available",
            "context_relevant": False
        }
    
    # Check context relevance
    context_relevance = chain.invoke(context_relevance_prompt.format(context=context, question=state["question"]))
    
    if context_relevance.lower() != "relevant":
        print("Context is not relevant to the question.")
        return {
            "answer": state["answer"],
            "docs": state["docs"],
            "is_relevant": False,
            "is_partially_relevant": False,
            "relevance_info": "Context not relevant to question",
            "context_relevant": False
        }
    
    # Check answer relevance
    answer_relevance = chain.invoke(answer_relevance_prompt.format(context=context, question=state["question"], answer=state["answer"]))
    
    is_relevant = answer_relevance.lower() == "relevant"
    is_partially_relevant = answer_relevance.lower() == "partially relevant"
    
    # Increment iteration count
    iteration_count = state.get("iteration_count", 0) + 1
    
    if iteration_count >= MAX_ITERATIONS:
        print(f"Reached maximum iterations ({MAX_ITERATIONS}). Ending search.")
        return {
            "answer": state["answer"],
            "docs": state["docs"],
            "is_relevant": True,  # Force end of search
            "is_partially_relevant": False,
            "relevance_info": "Max iterations reached",
            "context_relevant": True,
            "iteration_count": iteration_count
        }

    return {
        "answer": state["answer"],
        "docs": state["docs"],
        "is_relevant": is_relevant,
        "is_partially_relevant": is_partially_relevant,
        "relevance_info": answer_relevance,
        "context_relevant": True,
        "iteration_count": iteration_count
    }

def compute_semantic_similarity(sentence1, sentence2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Compute embeddings
    with torch.no_grad():
        embedding1 = sentence_model.encode([sentence1], convert_to_tensor=True).to(device)
        embedding2 = sentence_model.encode([sentence2], convert_to_tensor=True).to(device)
    
    # Compute cosine similarity
    similarity = 1 - cosine(embedding1[0].cpu().numpy(), embedding2[0].cpu().numpy())
    return similarity

# Function to rephrase the question
def rephrase_question(state):
    rephrase_count = state.get("rephrase_count", 0) + 1
    
    if rephrase_count > MAX_REPHRASE_ATTEMPTS:
        print(f"Reached maximum rephrase attempts ({MAX_REPHRASE_ATTEMPTS}). Ending search.")
        return {
            "question": state["question"],
            "rephrase_count": rephrase_count,
            "need_retrieval": False,
            "is_relevant": True  # Force end of search
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", REPHRASE_SYSTEM_PROMPT),
        ("human", "Original question: {question}\n\nPlease rephrase the question if necessary, do not ask for clarification if it's unclear:"),
    ])
    
    chain = prompt | ChatOpenAI(model=REPHRASE_QUESTION_MODEL) | StrOutputParser()
    
    # Generate rephrased question
    rephrased = chain.invoke({
        "question": state["question"]
    })

    # Check semantic similarity with original question
    similarity = compute_semantic_similarity(state["question"], rephrased)
    if similarity < SEMANTIC_SIMILARITY_THRESHOLD:
        print("Rephrased question is too different from the original. Keeping the original question.")
        return {
            "question": state["question"], 
            "rephrase_count": rephrase_count,
            "need_retrieval": False  # No need for retrieval if question didn't change
        }

    print(f"Rephrased question: {rephrased}")
    return {
        "question": rephrased, 
        "rephrase_count": rephrase_count,
        "need_retrieval": True  # Need retrieval if question changed
    }

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

def main(vectorstore: Any, question: str):
    workflow = StateGraph(State)

    # Add nodes individually
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("rephrase_question", rephrase_question)
    
    # Add edges
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
    
    app = workflow.compile()
    
    final_answer = None
    for output in app.stream({
        "question": question,
        "vectorstore": vectorstore,
        "docs": None,
        "answer": None,
        "is_relevant": False,
        "is_partially_relevant": False,
        "context_relevant": True,
        "relevance_info": None,
        "rephrase_count": 0,
        "iteration_count": 0,
        "need_retrieval": True  # Initially set to True
    }):
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "retrieve":
                    print(f"Current question: {value['question']}")
                    print(f"Retrieved {len(value['docs'])} documents")
                elif key == "generate_answer":
                    print(f"Generated answer: {value['answer']}")
                    final_answer = value['answer']
                elif key == "check_relevance":
                    print(f"Relevance check: {value['relevance_info']}")
                    if value['is_relevant']:
                        print("Answer is relevant. Ending search.")
                    elif value.get('is_partially_relevant', False):
                        print("Answer is partially relevant. Rephrasing question.")
                    elif not value['context_relevant']:
                        print("Context is not relevant. Rephrasing question.")
                    else:
                        print("Answer is not relevant. Retrieving new documents.")
                elif key == "rephrase_question":
                    print(f"Rephrased question: {value['question']}")
        elif output == END:
            break
    
    if final_answer:
        print(f"\nFinal answer: {final_answer}")
    else:
        print("No relevant information available after maximum iterations.")

if __name__ == "__main__":
    vectorstore = load_or_create_store(VECTOR_STORE_ROOT_FOLDER)

    question = "This is gibberish"
    main(vectorstore, question)