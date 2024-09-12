from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from constants import GENERATE_ANSWER_MODEL, CHECK_RELEVANCE_MODEL, CONTEXT_RELEVANCE_SYSTEM_PROMPT, ANSWER_RELEVANCE_SYSTEM_PROMPT, MAX_ITERATIONS

def generate_answer(state):
    """
    Generates an answer based on the given context and question.
    
    Args:
        state (dict): The current state of the workflow.
    
    Returns:
        dict: A dictionary containing the generated answer.
    """
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

def check_relevance(state):
    """
    Checks the relevance of the context and the generated answer.
    
    Args:
        state (dict): The current state of the workflow.
    
    Returns:
        dict: A dictionary containing relevance information and updated state.
    """
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
    
    answer_relevance = chain.invoke(answer_relevance_prompt.format(context=context, question=state["question"], answer=state["answer"]))
    
    is_relevant = answer_relevance.lower() == "relevant"
    is_partially_relevant = answer_relevance.lower() == "partially relevant"
    
    iteration_count = state.get("iteration_count", 0) + 1
    
    if iteration_count >= MAX_ITERATIONS:
        print(f"Reached maximum iterations ({MAX_ITERATIONS}). Ending search.")
        return {
            "answer": state["answer"],
            "docs": state["docs"],
            "is_relevant": True,
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
