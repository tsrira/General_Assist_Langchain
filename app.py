import torch
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Use a smaller, lighter model for memory optimization
MODEL_NAME = "distilgpt2"

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

CHUNK_SIZE = 400  # smaller chunks to reduce memory
CHUNK_OVERLAP = 80

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def embed_chunks(chunks):
    embedder = load_embedder()
    return embedder.encode(chunks, convert_to_numpy=True)

@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_length=256  # limit max tokens to reduce memory usage
    )
    return llm_pipe, tokenizer

def extract_and_chunk_pdf(pdf_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text.strip())
    full_text = "\n".join(texts)
    chunks = []
    start = 0
    length = len(full_text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def build_faiss_index(chunk_embeddings):
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)
    return index

def retrieve_relevant_chunks(query, embedder, index, chunks, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    similarities = 1 / (1 + distances[0])
    return [chunks[i] for i in indices[0]], max(similarities)

def get_handbook_response(question, llm_pipe, tokenizer, context, sim, threshold=0.4):
    if sim < threshold or not context.strip():
        return "Sorry! I can't find relevant info from the knowledge base."
    prompt = (
        f"You are a Student Handbook assistant. Generate answer ONLY from the context below.\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    )
    result = llm_pipe(
        prompt,
        do_sample=False,
        max_length=256,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    return result[0]['generated_text'][len(prompt):].strip()

st.title("Student Handbook Chatbot (Optimized)")

pdf_file = st.file_uploader("Upload your Student Handbook PDF", type=["pdf"])

if pdf_file:
    with st.spinner("Processing PDF..."):
        temp_path = "uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.read())
        chunks = extract_and_chunk_pdf(temp_path)

        st.write(f"Chunks created: {len(chunks)}")

        embedder = load_embedder()
        chunk_embeddings = embed_chunks(chunks)
        index = build_faiss_index(chunk_embeddings)

        llm_pipe, tokenizer = load_model()

    question = st.text_input("Ask your question:")

    if question:
        with st.spinner("Thinking..."):
            relevant_chunks, sim = retrieve_relevant_chunks(question, embedder, index, chunks)
            context = "\n\n".join(relevant_chunks)
            answer = get_handbook_response(question, llm_pipe, tokenizer, context, sim)
            st.markdown("**Answer:**")
            st.write(answer)
