# def load_vector_store(documents: List[Document]) -> VectorStore:
#     # TODO: properly save embeddings rather than recreating them every time
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL,
#         model_kwargs={"device": device})
#
#     db = FAISS.from_documents(documents, embeddings, distance_strategy=DistanceStrategy.COSINE)
#     return db
