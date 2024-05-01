from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever
import torch
import gradio as gr

def main():
    # Tokenizer and model from the specified model
    model_id = "LeoLM/leo-mistral-hessianai-7b-chat"
    tokenizer = RagTokenizer.from_pretrained(model_id)

    # Initialize RAG using the specified generator and retriever models
    model = RagTokenForGeneration.from_pretrained(
        model_id,
        retriever=RagRetriever.from_pretrained(
            model_id,
            index_name="custom",
            indexed_dataset=None,
            use_dummy_dataset=True,
            embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    )

    def generate_response(query, top_k=5):
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=top_k,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def chat_interface(query):
        response = generate_response(query)
        return response

    iface = gr.Interface(fn=chat_interface, inputs="text", outputs="text")
    iface.launch()

if __name__ == '__main__':
    main()
