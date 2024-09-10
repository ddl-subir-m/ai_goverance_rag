from constants import RETRIEVAL_TOP_K, RELEVANCE_THRESHOLD, RERANK_TOP_K
from ragatouille import RAGPretrainedModel

colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def retrieve(state):
    if not state["need_retrieval"]:
        print("Skipping retrieval as question hasn't changed.")
        return {"docs": state["docs"], "question": state["question"]}

    total_docs = state["vectorstore"]._collection.count()
    k = min(RETRIEVAL_TOP_K, total_docs)

    docs_and_scores = state["vectorstore"].similarity_search_with_relevance_scores(
        state["question"], 
        k=k
    )

    filtered_docs = [
        doc for doc, score in docs_and_scores 
        if score >= RELEVANCE_THRESHOLD
    ]

    print(f"Number of documents retrieved: {len(filtered_docs)}")
    
    if not filtered_docs:
        print("No documents retrieved. Returning empty list.")
        return {"docs": [], "question": state["question"]}
    
    rerank_k = min(RERANK_TOP_K, len(filtered_docs))
    
    reranked_docs = colbert_model.rerank(
        state["question"], 
        [doc.page_content for doc in filtered_docs], 
        k=rerank_k
    )
    
    formatted_docs = [
        {"content": doc["content"], "metadata": doc["metadata"] if "metadata" in doc else {}} 
        for doc in reranked_docs
    ]
    
    return {"docs": formatted_docs, "question": state["question"]}
