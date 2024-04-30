# General configuration parameters on each area


# Load text/documents
DATA_PATH = "sample_data/"

# Text embedding (sentence transformer)
# EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
# EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v1"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Vector Store
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "sample_docs"

# LLM (ollama)
LLMBASEURL = "http://localhost:11434/api"
# MODEL = 'phi3'
# MODEL = 'marco/em_german_mistral_v01'
MODEL = "LeoLM/leo-mistral-hessianai-7b-chat"


# Frontend
GUI_TITLE = f"Local RAG System ({MODEL})"
