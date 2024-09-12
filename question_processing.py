from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import torch
from constants import REPHRASE_SYSTEM_PROMPT, REPHRASE_QUESTION_MODEL, MAX_REPHRASE_ATTEMPTS, SEMANTIC_SIMILARITY_THRESHOLD, SENTENCE_TRANSFORMER_MODEL

sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

def compute_semantic_similarity(sentence1, sentence2):
    """
    Computes the semantic similarity between two sentences using a pre-trained model.
    
    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
    
    Returns:
        float: The semantic similarity score between the two sentences.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        embedding1 = sentence_model.encode([sentence1], convert_to_tensor=True).to(device)
        embedding2 = sentence_model.encode([sentence2], convert_to_tensor=True).to(device)
    
    similarity = 1 - cosine(embedding1[0].cpu().numpy(), embedding2[0].cpu().numpy())
    return similarity

def rephrase_question(state):
    """
    Rephrases the given question if necessary.
    
    Args:
        state (dict): The current state of the workflow.
    
    Returns:
        dict: A dictionary containing the rephrased question and updated state.
    """
    rephrase_count = state.get("rephrase_count", 0) + 1
    
    # Check if maximum rephrase attempts reached
    if rephrase_count > MAX_REPHRASE_ATTEMPTS:
        print(f"Reached maximum rephrase attempts ({MAX_REPHRASE_ATTEMPTS}). Ending search.")
        return {
            "question": state["question"],
            "rephrase_count": rephrase_count,
            "need_retrieval": False,
            "is_relevant": True
        }

    # Generate rephrased question
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPHRASE_SYSTEM_PROMPT),
        ("human", "Original question: {question}\n\nPlease rephrase the question if necessary, do not ask for clarification if it's unclear:"),
    ])
    
    chain = prompt | ChatOpenAI(model=REPHRASE_QUESTION_MODEL) | StrOutputParser()
    
    rephrased = chain.invoke({
        "question": state["question"]
    })

    # Check semantic similarity between original and rephrased questions
    similarity = compute_semantic_similarity(state["question"], rephrased)
    if similarity < SEMANTIC_SIMILARITY_THRESHOLD:
        print("Rephrased question is too different from the original. Keeping the original question.")
        return {
            "question": state["question"], 
            "rephrase_count": rephrase_count,
            "need_retrieval": False
        }

    print(f"Rephrased question: {rephrased}")
    return {
        "question": rephrased, 
        "rephrase_count": rephrase_count,
        "need_retrieval": True
    }


