# Vector store and retrieval settings
VECTOR_STORE_ROOT_FOLDER = "./synthetic_data"
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3

# Model settings
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-4" if you're using GPT-4

# RAG workflow settings
MAX_REPHRASE_ATTEMPTS = 3
SEMANTIC_SIMILARITY_THRESHOLD = 0.95

# Prompt templates
REPHRASE_SYSTEM_PROMPT = """You are an AI assistant tasked with rephrasing questions to improve information retrieval. Follow these guidelines:
1. Maintain the core intent of the original question.
2. Use more specific or technical terms if appropriate.
3. Break down complex questions into simpler components.
4. Add context from the previous answer if it was partially relevant.
5. Avoid introducing new concepts not present in the original question.
Your rephrased question should be clear, concise, and more likely to yield relevant information."""

CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You are an AI assistant tasked with determining if the given context is relevant to the question. Output 'relevant' if the context contains information related to the question, or 'not relevant' if it doesn't."""

ANSWER_RELEVANCE_SYSTEM_PROMPT = """You are an AI assistant tasked with determining if the given answer is relevant to the question and supported by the provided context. Consider the following criteria:
1. The answer directly addresses the question asked.
2. The information in the answer is present in or can be inferred from the given context.
3. The answer doesn't contain significant information that is not supported by the context.
Output 'relevant' if the answer meets these criteria, 'partially relevant' if it partially addresses the question but needs improvement, or 'not relevant' if it doesn't meet the criteria at all."""