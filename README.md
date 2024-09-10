# RAG-based Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) based question answering system using LangChain, LangGraph, and various NLP models.

## Project Overview

The system uses a combination of vector store retrieval, question rephrasing, and answer generation to provide accurate responses to user queries. It employs a workflow that iteratively improves the relevance of retrieved documents and generated answers.

## Key Components

1. Vector Store: Uses Chroma with CodeBERT embeddings for efficient document retrieval.
2. Retrieval: Implements a two-stage retrieval process with initial similarity search and re-ranking.
3. Answer Generation: Utilizes OpenAI's GPT models to generate answers based on retrieved contexts.
4. Relevance Checking: Assesses the relevance of both retrieved contexts and generated answers.
5. Question Rephrasing: Improves query effectiveness through intelligent question rephrasing.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/ddl-subir-m/ai_goverance_rag.git
   cd ai_goverance_rag
   ```

2. Set up environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Prepare your document corpus:
   Place your documents in the `./synthetic_data` directory (or modify the `VECTOR_STORE_ROOT_FOLDER` in `constants.py`).

2. Run the main script:
   ```
   python main.py
   ```

3. The script will create a vector store from your documents and then prompt you for questions. Enter your questions when prompted.

## Project Structure

- `main.py`: Entry point of the application.
- `workflow.py`: Defines the RAG workflow using LangGraph.
- `vector_store.py`: Handles creation and management of the vector store.
- `retrieval.py`: Implements document retrieval logic.
- `answer_generation.py`: Manages answer generation and relevance checking.
- `question_processing.py`: Handles question rephrasing.
- `embeddings.py`: Defines custom embedding models.
- `constants.py`: Contains configuration constants.
- `requirements.txt`: Contains dependencies for the project.

## Customization

- Modify `constants.py` to adjust model settings, thresholds, and other parameters.