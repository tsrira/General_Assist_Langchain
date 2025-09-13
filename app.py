import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline  # updated import
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate  # updated import
import numpy as np
from docx import Document  # pip install python-docx

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE_DEFAULT = 500
CHUNK_OVERLAP_DEFAULT = 50
RETRIEVER_TOP_K_DEFAULT = 3
SIMILARITY_THRESHOLD = 0.35

def read_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def read_doc_text(doc_file):
    doc = Document(doc_file)
    fullText = [para.text for para in doc.paragraphs]
    return "\n".join(fullText)

def extract_text_from_files(files):
    all_texts = []
    for file in files:
        filename = file.name.lower()
        if filename.endswith(".pdf"):
            text = read_pdf_text(file)
            all_texts.append(text)
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            text = read_doc_text(file)
            all_texts.append(text)
    return "\n\n".join(all_texts)

def chunk_document(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([text])
    return docs

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

def get_similarity(vectordb, embedder, query):
    query_embedding = embedder.embed_query(query)
    query_embedding = np.array(query_embedding)
    distances, _ = vectordb.index.search(query_embedding.reshape(1, -1), k=1)
    similarity = 1 / (1 + distances[0][0])
    return similarity

st.title("Student Handbook RAG Chatbot (LangChain)")

try:
    uploaded_files = st.file_uploader(
        "Upload your PDFs and DOC/DOCX files",
        type=["pdf", "doc", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner(f"Extracting and indexing {len(uploaded_files)} documents..."):
            combined_text = extract_text_from_files(uploaded_files)
            docs = chunk_document(combined_text, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT)
            embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.from_documents(docs, embedder)
            retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K_DEFAULT})
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

        st.subheader("Ask your question")
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("Thinking..."):
                similarity = get_similarity(vectordb, embedder, question)
                if similarity < SIMILARITY_THRESHOLD:
                    answer = "Sorry! I can't find relevant information from the knowledge base."
                else:
                    answer = qa_chain.invoke(question)  # use invoke() instead of run()
            st.markdown("**Chatbot:**")
            st.write(answer)
    else:
        st.info("Please upload one or more PDF or DOC/DOCX files to proceed.")

except Exception as e:
    st.error(f"An error occurred: {e}")
