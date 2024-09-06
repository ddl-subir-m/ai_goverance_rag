from dotenv import load_dotenv
import os
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ragatouille import RAGPretrainedModel
from langgraph.graph import StateGraph, END
from langchain.embeddings.base import Embeddings
import numpy as np
from typing import List

load_dotenv()
# Initialize ColBERT model
colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Function to ingest documents from a directory
from langchain_community.document_loaders import DirectoryLoader

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
    
    loader = DirectoryLoader(
        root_folder,
        recursive=True  
    )
    documents = loader.load()
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
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# Function to retrieve documents
def retrieve(state):
    retriever = ParentDocumentRetriever(
        vectorstore=state["vectorstore"],
        docstore=InMemoryStore(),
        search_kwargs={"k": 5},
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    )
    
    docs = retriever.get_relevant_documents(state["question"])
    reranked_docs = colbert_model.rerank(state["question"], [doc.page_content for doc in docs])
    
    return {"docs": reranked_docs[:3]}

# Function to generate an answer
def generate_answer(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the question based on the given context."),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ])
    
    chain = prompt | ChatOpenAI() | StrOutputParser()
    
    answer = chain.invoke({
        "context": "\n\n".join([doc["text"] for doc in state["docs"]]),
        "question": state["question"]
    })
    
    return {"answer": answer}

# Function to check answer relevance
def check_relevance(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine if the answer is relevant to the question. Output 'relevant' or 'not relevant'."),
        ("human", "Question: {question}\nAnswer: {answer}"),
    ])
    
    chain = prompt | ChatOpenAI() | StrOutputParser()
    
    result = chain.invoke({
        "question": state["question"],
        "answer": state["answer"]
    })
    
    return "rephrase" if result.lower() == "not relevant" else "end"

# Function to rephrase the question
def rephrase_question(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the given question to make it more relevant to the available information."),
        ("human", "Original question: {question}"),
    ])
    
    chain = prompt | ChatOpenAI() | StrOutputParser()
    
    rephrased = chain.invoke({"question": state["question"]})
    
    return {"question": rephrased}

# Main workflow
def main(root_folder: str, question: str):
    vectorstore = ingest_documents(root_folder)
    
    workflow = StateGraph(nodes=[retrieve, generate_answer, check_relevance, rephrase_question])
    
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "check_relevance")
    workflow.add_edge("check_relevance", "rephrase_question", condition=lambda x: x == "rephrase")
    workflow.add_edge("check_relevance", END, condition=lambda x: x == "end")
    workflow.add_edge("rephrase_question", "retrieve")
    
    workflow.set_entry_point("retrieve")
    
    app = workflow.compile()
    
    for output in app.stream({
        "question": question,
        "vectorstore": vectorstore,
    }):
        if "answer" in output:
            print(f"Answer: {output['answer']}")
        elif output.get("question") != question:
            print(f"Rephrased question: {output['question']}")
    
    if "answer" not in output:
        print("No relevant information available.")

if __name__ == "__main__":
    root_folder = "/path/to/your/documents"
    question = "Your question here"
    main(root_folder, question)