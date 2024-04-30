from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ragfuncs import (
    make_collection,
    get_collection,
    get_relevant_text,
    get_context_prompt,
    generate
)
from parameters import DATA_PATH, CHROMA_DATA_PATH, COLLECTION_NAME, MODEL
import os


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        use_cache=True
    )

    # make_collection(DATA_PATH, COLLECTION_NAME)
    collection = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME)

    print(f"\n============== Local RAG (Model: {MODEL}) ==============")
    print("(Press 'q' to quit)")
    while True:
        user_input = input("\nYour prompt: ")
        if user_input == 'q':
            break

        relevant_text = get_relevant_text(collection, user_input, sim_th=0.1)
        context_query = get_context_prompt(user_input, relevant_text)
        print('context_query: ')
        print(context_query)

        rag_response = generate(
            context_query,
            tokenizer=tokenizer,
            model=model,
            top_k=3,
            top_p=0.9,
            temp=0.3
        )

        print("Answer:")
        print(rag_response)

if __name__ == '__main__':
    main()
