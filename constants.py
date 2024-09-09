# Vector store and retrieval settings
VECTOR_STORE_ROOT_FOLDER = "./synthetic_data"
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3

# Model settings
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
OPENAI_MODEL = "gpt-4o-mini" 

# Node-specific model settings
GENERATE_ANSWER_MODEL ="gpt-4o-mini"
CHECK_RELEVANCE_MODEL = "gpt-4o-mini"
REPHRASE_QUESTION_MODEL = "gpt-4o-mini"

# RAG workflow settings
MAX_ITERATIONS = 3  # Maximum number of iterations for the entire workflow
MAX_REPHRASE_ATTEMPTS = 2
RELEVANCE_THRESHOLD = 0.7
SEMANTIC_SIMILARITY_THRESHOLD = 0.95

# Prompt templates
REPHRASE_SYSTEM_PROMPT = """You are an AI assistant tasked with rephrasing questions to improve information retrieval. Follow these guidelines:\n"
"1. Maintain the core intent and meaning of the original question.\n"
"2. Do not introduce new concepts or change the subject of the question.\n"
"3. If the original question is unclear or seems like gibberish, do not invent a new question. Instead, ask for clarification.\n"
"4. Make only minor adjustments to improve clarity if needed.\n"
"5. If the question is already clear and specific, return it unchanged.\n"
"Your rephrased question should be recognizably similar to the original."""

CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You are an AI assistant tasked with determining if the given context is relevant to the question. Output 'relevant' if the context contains information related to the question, or 'not relevant' if it doesn't."""

ANSWER_RELEVANCE_SYSTEM_PROMPT = """You are an AI assistant tasked with determining if the given answer is relevant to the question and supported by the provided context. Consider the following criteria:
1. The answer directly addresses the question asked.
2. The information in the answer is present in or can be inferred from the given context.
3. The answer doesn't contain significant information that is not supported by the context.
Output 'relevant' if the answer meets these criteria, 'partially relevant' if it partially addresses the question but needs improvement, or 'not relevant' if it doesn't meet the criteria at all."""