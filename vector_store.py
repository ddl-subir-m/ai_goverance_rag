from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from embeddings import CodeBERTEmbeddings

def create_vector_store(root_folder: str):
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
    
    codebert_embeddings = CodeBERTEmbeddings()
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=codebert_embeddings,
        collection_name="project_rag_collection",
    )

    print(f"Ingested {vectorstore._collection.count()} documents into Chroma")
    
    return vectorstore
