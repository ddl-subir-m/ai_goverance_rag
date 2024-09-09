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

# Initialize the chat model
chat_model = ChatOpenAI(model=OPENAI_MODEL)

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
    docs = state["vectorstore"].similarity_search(state["question"], k=RETRIEVAL_TOP_K)
    print(f"Number of documents retrieved: {len(docs)}")
    
    if not docs:
        print("No documents retrieved. Returning empty list.")
        return {"docs": []}
    
    rerank_k = min(RERANK_TOP_K, len(docs))
    
    reranked_docs = colbert_model.rerank(state["question"], [doc.page_content for doc in docs], k=rerank_k)
    
    # Convert reranked_docs to the expected format
    formatted_docs = [{"content": doc["content"], "metadata": doc["metadata"] if "metadata" in doc else {}} for doc in reranked_docs]
    
    return {"docs": formatted_docs}

# Function to generate an answer
def generate_answer(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the question based on the given context."),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ])
    
    chain = prompt | chat_model | StrOutputParser()
    
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
    
    chain = chat_model | StrOutputParser()
    
    context = "\n\n".join([doc["content"] for doc in state["docs"]]) if state["docs"] else ""
    
    if not context.strip():
        print("No context available. Marking as not relevant.")
        return {
            "answer": state["answer"],
            "docs": state["docs"],
            "is_relevant": False,
            "relevance_info": "No context available"
        }
    
    # Check context relevance
    context_relevance = chain.invoke(context_relevance_prompt.format(context=context, question=state["question"]))
    
    if context_relevance.lower() != "relevant":
        print("Context is not relevant to the question.")
        return {
            "answer": state["answer"],
            "docs": state["docs"],
            "is_relevant": False,
            "relevance_info": "Context not relevant to question"
        }
    
    # Check answer relevance
    answer_relevance = chain.invoke(answer_relevance_prompt.format(context=context, question=state["question"], answer=state["answer"]))
    
    is_relevant = answer_relevance.lower() == "relevant"
    is_partially_relevant = answer_relevance.lower() == "partially relevant"
    
    return {
        "answer": state["answer"],
        "docs": state["docs"],
        "is_relevant": is_relevant,
        "is_partially_relevant": is_partially_relevant,
        "relevance_info": answer_relevance
    }

def compute_semantic_similarity(sentence1, sentence2):
    # Compute embeddings
    embedding1 = sentence_model.encode([sentence1])[0]
    embedding2 = sentence_model.encode([sentence2])[0]
    
    # Compute cosine similarity
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# Function to rephrase the question
def rephrase_question(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPHRASE_SYSTEM_PROMPT),
        ("human", "Original question: {question}\n\nPrevious answer: {answer}\n\nContext: {context}\n\nPlease rephrase the question:"),
    ])
    
    chain = prompt | chat_model | StrOutputParser()
    
    context = "\n\n".join([doc["content"] for doc in state["docs"]]) if state["docs"] else ""
    
    # Check iteration count
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= MAX_REPHRASE_ATTEMPTS:
        print("Reached maximum rephrase attempts. Continuing with the current question.")
        return {"question": state["question"], "rephrase_count": rephrase_count + 1}

    # Generate rephrased question
    rephrased = chain.invoke({
        "question": state["question"],
        "answer": state["answer"] if state["answer"] else "No previous answer available.",
        "context": context
    })

    # Check semantic similarity with original question
    similarity = compute_semantic_similarity(state["question"], rephrased)
    if similarity > SEMANTIC_SIMILARITY_THRESHOLD:
        print("Rephrased question is very similar to the original. Keeping the original question.")
        return {"question": state["question"], "rephrase_count": rephrase_count + 1}

    print(f"Rephrased question: {rephrased}")
    return {
        "question": rephrased, 
        "rephrase_count": rephrase_count + 1
    }

class State(TypedDict):
    question: str
    vectorstore: Any
    docs: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    is_relevant: bool
    is_partially_relevant: bool
    relevance_info: Optional[str]
    rephrase_count: int

# Main workflow
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
        lambda x: "end" if x["is_relevant"] else "rephrase" if x.get("is_partially_relevant") else "retrieve",
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
        "relevance_info": None,
        "rephrase_count": 0
    }):
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "retrieve":
                    print(f"Retrieved {len(value['docs'])} documents")
                elif key == "generate_answer":
                    print(f"Generated answer: {value['answer']}")
                    final_answer = value['answer']
                elif key == "check_relevance":
                    print(f"Relevance check: {value['relevance_info']}")
                    if value['is_relevant']:
                        print("Answer is relevant. Ending search.")
                    elif value.get('is_partially_relevant'):
                        print("Answer is partially relevant. Rephrasing question.")
                    else:
                        print("Answer or context is not relevant. Retrieving new documents.")
                elif key == "rephrase_question":
                    print(f"Rephrased question: {value['question']}")
        elif output == END:
            break
    
    if final_answer:
        print(f"\nFinal answer: {final_answer}")
    else:
        print("No relevant information available.")

if __name__ == "__main__":
    vectorstore = load_or_create_store(VECTOR_STORE_ROOT_FOLDER)

    question = "This is gibberish"
    main(vectorstore, question)