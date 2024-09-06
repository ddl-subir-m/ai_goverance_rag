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
from langchain_community.vectorstores import Chroma
from ragatouille import RAGPretrainedModel
from langgraph.graph import StateGraph, END

# Initialize ColBERT model
colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Function to ingest documents from a directory
from langchain_community.document_loaders import DirectoryLoader

def ingest_documents(root_folder: str):
    # Define an exhaustive list of file extensions for data science projects
    ds_exts = [
        # Code files
        "*.py", "*.ipynb", "*.r", "*.rmd", "*.jl", "*.scala", "*.sql", "*.sas",
        # Data files
        "*.csv", "*.tsv", "*.json", "*.jsonl", "*.xml", "*.parquet", "*.avro",
        "*.orc", "*.feather", "*.arrow", "*.hdf5", "*.h5", "*.pkl", "*.pickle",
        # Config and metadata files
        "*.yml", "*.yaml", "*.toml", "*.ini", "*.cfg", "*.conf",
        # Documentation and text files
        "*.md", "*.txt", "*.rst", "*.tex", "*.pdf",
        # Spreadsheets (note: might require additional processing)
        "*.xlsx", "*.xls", "*.ods",
        # Other potentially relevant files
        "*.log", "*.sh", "*.bat", "*.ps1", "*.dockerfile", "*.proto"
    ]
    
    # Create a loader with the specified extensions
    loader = DirectoryLoader(
        root_folder,
        glob="**/{" + ",".join(ds_exts) + "}",
        recursive=True
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(splits, colbert_model.embed_documents)
    
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