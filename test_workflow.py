import nltk
import os
from rag_workflow import main, ingest_documents

def create_synthetic_data(root_folder: str):
    os.makedirs(root_folder, exist_ok=True)
    
    # Sample Python file
    with open(os.path.join(root_folder, "data_processing.py"), "w") as f:
        f.write("""
def process_data(data):
    # This function processes incoming data
    cleaned_data = [item.strip().lower() for item in data if item]
    return cleaned_data

def analyze_results(results):
    # This function analyzes the results of data processing
    total = sum(results)
    average = total / len(results) if results else 0
    return {"total": total, "average": average}
""")
    
    # Sample Markdown file
    with open(os.path.join(root_folder, "project_overview.md"), "w") as f:
        f.write("""
# Data Science Project Overview

This project aims to analyze large datasets using advanced machine learning techniques.

## Key Components:
1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Evaluation and Deployment

Our primary focus is on improving prediction accuracy while maintaining computational efficiency.
""")
    

def test_workflow():
    root_folder = "./synthetic_data"
    create_synthetic_data(root_folder)
    
    # Test document ingestion
    vectorstore = ingest_documents(root_folder)
    print(f"Ingested {vectorstore._collection.count()} documents")
    
    # Test the main workflow with a sample question
    question = "What are the key components of the data science project?"
    main(root_folder, question)
    
    # Additional test questions
    # test_questions = [
    #     "How does the process_data function work?",
    #     "What is the learning rate in the model parameters?",
    #     "Where are the train and test data stored?"
    # ]
    
    # for q in test_questions:
    #     print(f"\nTesting question: {q}")
    #     main(root_folder, q)

if __name__ == "__main__":
    nltk.download('punkt', download_dir='./.venv/nltk_data')
    nltk.download('punkt_tab', download_dir='./.venv/nltk_data')
    nltk.download('averaged_perceptron_tagger_eng', download_dir='./.venv/nltk_data')
    test_workflow()