from utils import list_files, read_file, get_chunks
import os
import chromadb
from chromadb.utils import embedding_functions
import requests, json, random
from parameters import EMBEDDING_MODEL, CHROMA_DATA_PATH
from parameters import LLMBASEURL, MODEL
from load_csv import load_csv
import torch


def make_collection(data_path, collection_name, skip_included_files=True):
    """Create vector store collection from a set of documents"""

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    files = list_files(data_path, extensions=('.csv'))
    print(f"Embedding files: {', '.join(files)} ...")

    if skip_included_files:
        sources = {m.get('source') for m in collection.get().get('metadatas')}

    for f in files:
        _, file_name = os.path.split(f)

        # if skip_included_files and file_name in sources:
        #     print(file_name, "already in Vector-DB, skipping...")
        #     continue
        #
        # text = read_file(f)

        print(f"Getting chunks for {file_name} ...")
        # chunks = get_chunks(text)
        chunks = load_csv()

        print(f"Embedding and storing {file_name} ...")
        collection.add(
            documents=chunks,
            ids=[f"id{file_name[:-4]}.{j}" for j in range(len(chunks))],
            metadatas=[{"source": file_name, "part": n} for n in range(len(chunks))],
        )


def get_collection(vector_store_path, collection_name):
    """Load a saved vector store collection"""

    print(f"Loading collection {collection_name} ...")
    client = chromadb.PersistentClient(path=vector_store_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(name=collection_name, embedding_function=embedding_func)

    return collection


def get_relevant_text(collection, query='', nresults=2, sim_th=0.5):
    """Get relevant text from a collection for a given query"""

    query_result = collection.query(query_texts=query, n_results=nresults)
    docs = query_result.get('documents')[0]
    if sim_th is not None:
        similarities = [1 - d for d in query_result.get("distances")[0]]
        relevant_docs = [f"similarity: {round(s,2)}: {d}" for d, s in zip(docs, similarities) if s >= sim_th]
        return '\n'.join(relevant_docs)
    return ''.join(docs)


# LLM Funcs (Ollama)
def generate(prompt, tokenizer, model, top_k=5, top_p=0.9, temp=0.2):
    # Tokenize the input prompt for the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # Generate a response using the model
    # response = model.generate(
    #     **inputs,
    #     max_length=500,  # Adjust as necessary
    #     num_return_sequences=1,
    #     # temperature=temp,
    #     # top_p=top_p,
    #     # top_k=top_k,
    #     # no_repeat_ngram_size=2,  # Optional: to avoid repetitive text
    # )
    with torch.no_grad():
        # response = model.generate(**inputs, max_length=1024)
        response = model.generate(
                **inputs,
                max_length=1024,  # Adjust as necessary
                # num_return_sequences=1,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                # no_repeat_ngram_size=2,  # Optional: to avoid repetitive text
                do_sample=True
            )

    # Decode the generated ids to text
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
    print(generated_text)
    answer = generated_text.split("\n")[-2]
    print('answer: ')
    print(answer)
    return answer



def llm_mockup(prompt, top_k=1, top_p=0.9, temp=0.5):
    return random.choice(["Yes!", "Not sure", "It depends", "42"])


def get_context_prompt(question, context):

    contextual_prompt = (
        "Nutze folgenden Kontext, um die Frage am Ende zu beantworten.\n"
        "Nutze ausschließlich Informationen aus dem Kontext.\n"
        "Hinter 'Begriffe für Laientext:' stehen Ausdrücke, die im Laientext verwendet werden sollen.\n"
        "Hinter 'Begriffe für Profitext:' stehen Ausdrücke, die im Profitext verwendet werden sollen.\n"
        "Hinter 'vermeiden beim Laien:' stehen Ausdrücke, die im Laientext vermieden werden sollen.\n"
        # "Fasse Dich in Deiner Antwort möglichst kurz und präzise.\n"
        "Kontext:\n"
        f"{context}\n"
        "\nFrage:\n"
        f"{question}"
    )

    return contextual_prompt



# if __name__ == "__main__":
#     # Quick RAG sample check
#     from parameters import DATA_PATH, COLLECTION_NAME
#
#     make_collection(DATA_PATH, COLLECTION_NAME)
#
#     collection = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME)
#
#     # Query
#     query = "Where can I learn about artificial intelligence in Berlin?"
#     # query = "What happened to John McClane in Christmas?"
#     # query = "Who is Sherlock Holmes?"
#
#     print("\nQuery:", query)
#
#     relevant_text = get_relevant_text(collection, query)
#
#     print("\nRelevant text:")
#     print(relevant_text)
#
#     # LLM Cli
#     print("\nQuering LLM...")
#     context_query = get_context_prompt(query, relevant_text)
#     bot_response = generate(context_query)
#
#     print("\nModel Answer:")
#     print(bot_response)

if __name__ == '__main__':
    from parameters import CHROMA_DATA_PATH, COLLECTION_NAME, MODEL
    collection = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME)

    text = get_relevant_text(collection, 'Was ist das Abdomen?')
    print(text)
