# General configuration parameters on each area


# Load text/documents
DATA_PATH = "sample_data/"

# Text embedding (sentence transformer)
EMBEDDING_MODEL_THREE = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_MODEL_ONE = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL_TWO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL_THREE = "distiluse-base-multilingual-cased-v1"
# EMBEDDING_MODEL = "danielheinz/e5-base-sts-en-de"

# Vector Store
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME_ONE = "sample_docs_1"
COLLECTION_NAME_TWO = "sample_docs_2"
COLLECTION_NAME_THREE = "sample_docs_3"

# LLM (ollama)
LLMBASEURL = "http://localhost:11434/api"
# MODEL = 'phi3'
# MODEL = 'marco/em_german_mistral_v01'
# MODEL = "LeoLM/leo-mistral-hessianai-7b-chat"
MODEL = "LeoLM/leo-hessianai-7b-chat"


# Frontend
GUI_TITLE = f"Local RAG System ({MODEL})"
