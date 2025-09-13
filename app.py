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
from docx import Document  # pip install python-docx

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE_DEFAULT = 500
CHUNK_OVERLAP_DEFAULT = 50
RETRIEVER_TOP_K_DEFAULT = 3
SIMILARITY_THRESHOLD = 0.35

# Folder path containing documents
DOCUMENTS_FOLDER = st.text_input("Enter the folder path containing PDFs and DOC files:")

def read_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def read_doc_text(doc_path):
    doc = Document(doc_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)

def extract_text_from_folder(folder_path):
    all_texts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            text = read_pdf_text(filepath)
            all_texts.append(text)
        elif filename.lower().endswith(".docx") or filename.lower().endswith(".doc"):
            text = read_doc_text(filepath)
            all_texts.append(text)
    return "\n\n".join(all_texts)

def chunk_document(text, chunk_size, chunk_overlap):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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

with st.spinner(f"Loading and indexing documents from folder: {DOCUMENTS_FOLDER} ..."):
    combined_text = extract_text_from_folder(DOCUMENTS_FOLDER)
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

st.subheader("Ask about the Student Handbook, Indian Veg Recipe, Onboarding or Panicker Travels")
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

