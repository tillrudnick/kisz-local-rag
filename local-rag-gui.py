from ragfuncs import (
    get_collection,
    make_collection,
    get_relevant_text,
    get_context_prompt,
    generate,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from parameters import DATA_PATH, COLLECTION_NAME_TWO, COLLECTION_NAME_THREE, EMBEDDING_MODEL_ONE, EMBEDDING_MODEL_TWO, \
    EMBEDDING_MODEL_THREE
from parameters import CHROMA_DATA_PATH, COLLECTION_NAME_ONE, GUI_TITLE, MODEL
import gradio as gr


def main():
    make_collection(DATA_PATH, COLLECTION_NAME_ONE, EMBEDDING_MODEL_ONE)
    make_collection(DATA_PATH, COLLECTION_NAME_TWO, EMBEDDING_MODEL_TWO)
    make_collection(DATA_PATH, COLLECTION_NAME_THREE, EMBEDDING_MODEL_THREE)
    collection_one = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME_ONE, EMBEDDING_MODEL_ONE)
    collection_two = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME_TWO, EMBEDDING_MODEL_TWO)
    collection_three = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME_THREE, EMBEDDING_MODEL_THREE)
    model_id = MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        device_map="auto",
        use_flash_attention_2=True
    )

    def rag(user_msg, history, top_k, top_p, temp):
        relevant_text_one = get_relevant_text(collection_one, user_msg, sim_th=0.4)
        relevant_text_two = get_relevant_text(collection_two, user_msg, sim_th=0.4)
        relevant_text_three = get_relevant_text(collection_three, user_msg, sim_th=0.4)
        context_query = get_context_prompt(user_msg, relevant_text_one)
        bot_response = generate(context_query, tokenizer, model, top_k=top_k, top_p=top_p, temp=temp)
        print(f"{EMBEDDING_MODEL_ONE}: {relevant_text_one}\n\n")
        print(f"{EMBEDDING_MODEL_TWO}: {relevant_text_two}\n\n")
        print(f"{EMBEDDING_MODEL_THREE}: {relevant_text_three}\n\n")
        return f"Kontext:\n {relevant_text_one}\n\nAntwort: {bot_response}"


    chatgui = gr.ChatInterface(
        rag,
        title=GUI_TITLE,
        chatbot=gr.Chatbot(height=700),
        additional_inputs=[
            gr.Slider(1, 10, value=1, step=1, label="Top k"),
            gr.Slider(0.1, 1, value=0.1, step=0.1, label="Top p"),
            gr.Slider(0.1, 1, value=0.1, step=0.1, label="Temp"),
        ],
    )
    chatgui.launch()


if __name__ == '__main__':
    main()
