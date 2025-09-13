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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SIMILARITY_THRESHOLD = 0.5  # Adjust based on testing

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

def chunk_document(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text])
    return docs

def read_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def get_similarity(vectordb, embedder, query):
    query_embedding = embedder.embed_query(query)
    distances, _ = vectordb.index.search(query_embedding.reshape(1, -1), k=1)
    similarity = 1 / (1 + distances[0][0])
    return similarity

st.title("Student Handbook RAG Chatbot (LangChain)")

pdf_file = st.file_uploader("Upload your Student Handbook PDF", type=["pdf"])

if pdf_file:
    with st.spinner("Extracting and indexing PDF..."):
        text = read_pdf_text(pdf_file)
        docs = chunk_document(text)
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embedder)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = load_llm()

        template = """
        You are a helpful assistant. Answer the question ONLY based on the context below.
        If you do not find relevant information, respond with exactly:
        "Sorry, I do not have that information."

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

    st.subheader("Ask about the Handbook")
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Thinking..."):
            similarity = get_similarity(vectordb, embedder, question)
            if similarity < SIMILARITY_THRESHOLD:
                answer = "Sorry, I do not have that information."
            else:
                answer = qa_chain.run(question)
            st.markdown("**Chatbot:**")
            st.write(answer)
