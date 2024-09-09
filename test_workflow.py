import nltk
import os
from rag_workflow import load_or_create_store, main

def create_synthetic_data(root_folder: str):
    # Check if directory exists before creating
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    # Sample Python file
    file_path = os.path.join(root_folder, "data_processing.py")
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
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
    
    # Load or create store
    vectorstore = load_or_create_store(root_folder)
    
    # Test the main workflow with a sample question
    question = "What functions are used here?"
    main(vectorstore, question)
    
    # Additional test questions
    test_questions = [
        "How does the process_data function work?",
        # "What are the key components of the data science project?",
        # "What is the primary focus of the project?"
    ]
    
    for q in test_questions:
        print(f"\nTesting question: {q}")
        main(vectorstore, q)

def ensure_nltk_packages():
    nltk_data_dir = './.venv/nltk_data'
    packages = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng']

    for package in packages:
        try:
            nltk.data.find(f'{package}', paths=[nltk_data_dir])
            print(f"{package} is already up to date.")
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)

if __name__ == "__main__":
    ensure_nltk_packages()
    test_workflow()

