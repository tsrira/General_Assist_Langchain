import streamlit as st
import pdfplumber
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import numpy as np

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE_DEFAULT = 500
CHUNK_OVERLAP_DEFAULT = 50
RETRIEVER_TOP_K_DEFAULT = 3
SIMILARITY_THRESHOLD = 0.35

def chunk_pdf_in_batches(pdf_file, chunk_size, chunk_overlap):
    docs = []
    with pdfplumber.open(pdf_file) as pdf:
        batch_size = 5  # pages per batch
        for i in range(0, len(pdf.pages), batch_size):
            batch_text = "\n".join(page.extract_text() or "" for page in pdf.pages[i:i+batch_size])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            batch_docs = splitter.create_documents([batch_text])
            docs.extend(batch_docs)
    return docs

def read_doc_text(doc_file):
    doc = Document(doc_file)
    fullText = [para.text for para in doc.paragraphs]
    return "\n".join(fullText)

def chunk_document(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([text])
    return docs

def extract_and_chunk_files(files, chunk_size, chunk_overlap):
    all_docs = []
    for file in files:
        filename = file.name.lower()
        if filename.endswith(".pdf"):
            pdf_docs = chunk_pdf_in_batches(file, chunk_size, chunk_overlap)
            all_docs.extend(pdf_docs)
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            text = read_doc_text(file)
            docs = chunk_document(text, chunk_size, chunk_overlap)
            all_docs.extend(docs)
    return all_docs

@st.cache_resource(show_spinner=False)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN, trust_remote_code=True)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

#@st.cache_resource(show_spinner=False)
def create_vectordb(_docs):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(_docs, embedder)
    return vectordb, embedder

def get_similarity(vectordb, embedder, query):
    query_embedding = embedder.embed_query(query)
    query_embedding = np.array(query_embedding)
    distances, _ = vectordb.index.search(query_embedding.reshape(1, -1), k=1)
    similarity = 1 / (1 + distances[0][0])
    return similarity

st.title("General RAG Chatbot (LangChain)")

uploaded_files = st.file_uploader(
    "Upload your PDFs and DOC/DOCX files",
    type=["pdf", "doc", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner(f"Extracting and indexing {len(uploaded_files)} documents..."):
        all_docs = extract_and_chunk_files(uploaded_files, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT)
        vectordb, embedder = create_vectordb(all_docs)
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
                answer = qa_chain.invoke(question)

        st.markdown("**Chatbot:**")
        st.write(answer)
else:
    st.info("Please upload one or more PDF or DOC/DOCX files to proceed.")




