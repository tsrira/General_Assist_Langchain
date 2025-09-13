# 🦙 Student Handbook RAG Chatbot (LangChain) 📖🤖

Welcome to a fully open-source, retrieval-augmented chatbot powered by (LangChain) and Streamlit. This app allows you to query university handbooks or similar documents and get concise, context-aware answers—never hallucinating, always grounded in your uploaded PDF!

<div align="center">
  <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" height="60" />
</div>

## 🚀 Features

- **Upload any student handbook PDF**: Instantly chunked and indexed for semantic search.
- **Ask questions naturally**: Retrieval-Augmented Generation (RAG) ensures answers are grounded in your specific document content.
- **Powered by Mistral-7B**: High-quality generative language model handles query answering.
- **Efficient embedding & retrieval**: Leverages Sentence Transformers & FAISS for fast similarity search.
- **Modern Streamlit UI**: Clean and interactive user experience.

## 💡 Usage
Step 1: Click "Upload your Student Handbook PDF" on the sidebar.

Step 2: Type your questions about policies, exams, support, grading, and more!

Step 3: The chatbot will return answers that are strictly grounded in the context of your uploaded document. If the information doesn’t exist, it refuses politely.

## 📦 Technologies Used
Streamlit

Transformers

LangChain

google/flan-t5-base

HuggingFace

Sentence Transformers

FAISS

pdfplumber

bitsandbytes

## 🔒 Guardrails & Reliability
Retrieval-augmented: No out-of-context hallucination. Answers are solely based on semantic similarity to user queries and actual PDF content.

Strict refusal: The bot returns a polite message if the answer is not present in the uploaded knowledge base.

## 📝 Example
User: "What is the attendance requirement?"

Chatbot: "UG (nonengineering): 75% per subject. Engineering: 75% per semester."

User: "Who is the President of the United States?"

Chatbot: "Sorry! I can't find relevant information from the knowledge base. Don't provide any additional information."

## 📢 Contributing
Pull requests, issues, and discussions are welcome! Feel free to fork the repository and improve the app for other handbooks or documents.


## 📸 Screenshot

![App Screenshot](./images/screenshot.png)

## 🎓 About

This chatbot is built for students to easily query university policies, grading schemes, exam formats, and much more—directly from the official PDF handbook.

## 📝 License

MIT License © 2025 YourName

## 🤝 Contributions

Contributions and feedback are welcome! Please open an issue or pull request for improvements.

---

Made with ❤️ and Mistral 🚀
