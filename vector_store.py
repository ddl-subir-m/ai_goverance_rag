from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from embeddings import CodeBERTEmbeddings

def create_vector_store(root_folder: str):
    """
    Create a vector store from documents in the specified folder.

    Args:
        root_folder (str): The path to the folder containing the documents.

    Returns:
        Chroma: A Chroma vector store containing the embedded documents.
    """
    print(f"Loading documents from: {root_folder}")
    
    # Load documents from the specified folder
    loader = DirectoryLoader(root_folder, recursive=True)
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")
    
    if not documents:
        print("No documents were loaded. Check the root_folder path and file contents.")
        return None
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Number of document splits: {len(splits)}")
    
    # Create CodeBERT embeddings
    codebert_embeddings = CodeBERTEmbeddings()
    
    # Create and populate the Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=codebert_embeddings,
        collection_name="project_rag_collection",
    )

    print(f"Ingested {vectorstore._collection.count()} documents into Chroma")
    
    return vectorstore
