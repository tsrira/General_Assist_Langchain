import torch
import os
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def embed_chunks(chunks):
    embedder = load_embedder()
    return embedder.encode(chunks, convert_to_numpy=True)


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


def extract_and_chunk_pdf(pdf_path, chunk_size=500, chunk_overlap=50):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text.strip())
    full_text = "\n".join(texts)
    chunks = []
    start = 0
    text_len = len(full_text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def build_faiss_index(chunk_embeddings):
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    return index


def retrieve_relevant_chunks(query, embedder, index, chunks, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    similarities = 1 / (1 + distances[0])  # simple similarity scoring
    return [chunks[i] for i in indices[0]], max(similarities)


def get_handbook_response(question, llm_pipe, tokenizer, context, sim, threshold=0.4, max_length=512):
    if sim < threshold or not context.strip():
        return "Sorry! I can't find relevant information from the knowledge base. Don't provide any additional information."
    prompt_template = (
        "You are a Student Handbook assistant. You Should Generate and summarize the answer only from the given context below.\n"
        "If context is irrelevant to the question, You should say, Sorry! I can't find relevant information from the knowledge base. Don't provide any additional information.\n"
        "You should not generate or interpret any response from your knowledge. You are a helpful assistant.\n\n"
        "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question + "\nANSWER:"
    )
    generated = llm_pipe(
        prompt_template,
        do_sample=False,
        num_return_sequences=1,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = generated[0]['generated_text'][len(prompt_template):].strip()
    return answer


st.title("Student Handbook RAG Chatbot (Mistral-7B)")

pdf_file = st.file_uploader("Upload your Student Handbook PDF", type=["pdf"])

if pdf_file:
    with st.spinner("Processing PDF..."):
        temp_path = "uploaded_handbook.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.read())
        chunks = extract_and_chunk_pdf(temp_path, CHUNK_SIZE, CHUNK_OVERLAP)
        st.write(f"Total chunks created: {len(chunks)}")

        embedder = load_embedder()
        chunk_embeddings = embed_chunks(chunks)
        index = build_faiss_index(chunk_embeddings)
        llm_pipe, tokenizer = load_model()

    st.subheader("Ask about the Handbook")
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Retrieving answer..."):
            relevant_chunks, sim = retrieve_relevant_chunks(question, embedder, index, chunks)
            context = "\n\n".join(relevant_chunks)
            answer = get_handbook_response(question, llm_pipe, tokenizer, context, sim)

            st.markdown("**Chatbot:**")
            st.write(answer)


