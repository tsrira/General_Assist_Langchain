import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import numpy as np

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE_DEFAULT = 500
CHUNK_OVERLAP_DEFAULT = 50
RETRIEVER_TOP_K_DEFAULT = 3

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

def chunk_document(text, chunk_size, chunk_overlap):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text])
    return docs

def read_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def get_similarity(vectordb, embedder, query):
    query_embedding = embedder.embed_query(query)
    query_embedding = np.array(query_embedding)
    distances, _ = vectordb.index.search(query_embedding.reshape(1, -1), k=1)
    similarity = 1 / (1 + distances[0][0])
    return similarity

def suggest_similarity_threshold(vectordb, embedder, in_context_queries, out_context_queries):
    in_scores = [get_similarity(vectordb, embedder, q) for q in in_context_queries]
    out_scores = [get_similarity(vectordb, embedder, q) for q in out_context_queries]

    max_out = max(out_scores) if out_scores else 0.0
    min_in = min(in_scores) if in_scores else 1.0

    suggested_threshold = (max_out + min_in) / 2

    return {
        "in_context_scores": in_scores,
        "out_context_scores": out_scores,
        "max_out_context_similarity": max_out,
        "min_in_context_similarity": min_in,
        "suggested_threshold": suggested_threshold
    }

st.title("Student Handbook RAG Chatbot (LangChain)")

# Sidebar controls for tuning

# CHUNK_SIZE = st.sidebar.slider("Chunk Size", 300, 800, CHUNK_SIZE_DEFAULT, step=50)
# CHUNK_OVERLAP = st.sidebar.slider("Chunk Overlap", 20, 150, CHUNK_OVERLAP_DEFAULT, step=10)
# RETRIEVER_TOP_K = st.sidebar.slider("Retriever Top K", 1, 7, RETRIEVER_TOP_K_DEFAULT, step=1)
# SIMILARITY_THRESHOLD = st.sidebar.slider("Similarity Threshold", 0.1, 0.7, 0.32, step=0.01)

CHUNK_SIZE = CHUNK_SIZE_DEFAULT
CHUNK_OVERLAP = CHUNK_OVERLAP_DEFAULT
RETRIEVER_TOP_K = RETRIEVER_TOP_K_DEFAULT
SIMILARITY_THRESHOLD = 0.50

pdf_file = st.file_uploader("Upload your Student Handbook PDF", type=["pdf"])

if pdf_file:
    with st.spinner("Extracting and indexing PDF..."):
        text = read_pdf_text(pdf_file)
        docs = chunk_document(text, CHUNK_SIZE, CHUNK_OVERLAP)
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embedder)
        retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
        llm = load_llm()

        template = """
        You are a helpful assistant. Answer the question ONLY based on the context below.
        If you do not find relevant information, respond with exactly:
        "Sorry! I can't find relevant information from the knowledge base."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
        )

    # st.subheader("Sample similarity scores for tuning")
    in_context_examples = [
        "What is the attendance policy?",
        "How do I apply for leave?",
        "Describe the grading system."
    ]
    out_context_examples = [
        "Who is Donald Trump?",
        "What is the capital of France?",
        "Tell me about quantum computers."
    ]

    results = suggest_similarity_threshold(vectordb, embedder, in_context_examples, out_context_examples)
    
    # st.write(f"Similarities (in-context): {[f'{s:.3f}' for s in results['in_context_scores']]}")
    # st.write(f"Similarities (out-of-context): {[f'{s:.3f}' for s in results['out_context_scores']]}")
    # st.write(f"Max out-of-context similarity: {results['max_out_context_similarity']:.3f}")
    # st.write(f"Min in-context similarity: {results['min_in_context_similarity']:.3f}")
    # st.write(f"Suggested similarity threshold: {results['suggested_threshold']:.3f}")

    st.subheader("Ask about the Handbook")
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Thinking..."):
            similarity = get_similarity(vectordb, embedder, question)
            if similarity < SIMILARITY_THRESHOLD:
                answer = "Sorry! I can't find relevant information from the knowledge base."
            else:
                answer = qa_chain.run(question)
            st.markdown("**Chatbot:**")
            st.write(answer)









