import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODEL_NAME = "distilgpt2"

@st.cache_resource
def load_model():
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        llm_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        return llm_pipe, tokenizer
    except ImportError as e:
        # Fall back if bitsandbytes fails due to CUDA missing
        st.warning(
            "BitsAndBytes CUDA not available; falling back to full precision model."
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        llm_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        return llm_pipe, tokenizer

