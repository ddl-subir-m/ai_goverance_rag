from vector_store import create_vector_store
from workflow import setup_workflow
from constants import VECTOR_STORE_ROOT_FOLDER
from langgraph.graph import END

def main(app, vectorstore, question: str):
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
        "need_retrieval": True
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
    vectorstore = create_vector_store(VECTOR_STORE_ROOT_FOLDER)
    app = setup_workflow()

    question = "This is gibberish"
    main(app, vectorstore, question)