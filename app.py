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

# Your helper functions chunk_pdf_in_batches, read_doc_text, etc. (not shown here for brevity)
# Ensure all these functions are correctly indented


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
