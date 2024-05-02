import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
from parameters import DATA_PATH, CHROMA_DATA_PATH, COLLECTION_NAME_ONE, MODEL
from langchain.llms.ctransformers import CTransformers
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

def load_llm():
    # optimization only works with ampere GPUs!
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Regular LeoLM Mistral Chat Model
    model_id = MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        use_flash_attention_2=True
    )
    # hacky: suppress warning spam about unset pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    pipeline = transformers.pipeline(
        # model and tokenizer parameters
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # basic generation settings
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # TODO: bug! only can do max_length=4096 (sliding window length) with cache enabled.
        #  without cache max_length=8192 (or up to 128k) is possible.
        max_length=8192,
        use_cache=False,
        # TODO: advanced generation settings. play around with them.
        do_sample=True,
        typical_p=0.85,
        top_p=0.85,
        top_k=50,
        temperature=0.7,
    )
    llm = HuggingFacePipeline(model_id=model_id, pipeline=pipeline)
    return llm
