# 🌌 RAG TERMINAL

An advanced AI Research Assistant that transforms static PDF documents into an interactive intelligence base. This project uses **Retrieval-Augmented Generation (RAG)** to provide accurate, cited answers from your uploaded files.

---

## 🚀 Live Demo
[**Click here to view the App on Hugging Face**](https://huggingface.co/spaces/Zoha-Anjum/RAG_Chatbot)

---

## ✨ Key Features
- **Semantic Understanding:** Uses `Sentence-Transformers` to understand the meaning of your questions, not just keywords.
- **Conversational Memory:** Remembers the context of your chat, allowing for natural follow-up questions.
- **Verified Citations:** Every answer includes **Page Numbers** from the source PDF to ensure accuracy and eliminate hallucinations.
- **Neon UI:** A modern, user-friendly "Terminal" interface built with custom CSS.
- **Data Export:** Download your entire research session as a `.txt` file with one click.
- **Smart Chunking:** Uses `RecursiveCharacterTextSplitter` to maintain document context.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **LLM:** Llama-3.1-8b-instant (via Groq API)
- **Orchestration:** LangChain Classic
- **Vector Database:** FAISS
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **UI Framework:** Gradio 5.x

---

## 📖 How to Use
1. **Upload:** Drag and drop your PDF files into the **Data Injection** sidebar.
2. **Initialize:** Click the **🚀 INITIALIZE SYSTEM** button. 
3. **Status:** Wait for the system status to turn **🟢 Online**.
4. **Chat:** Ask questions about your documents in the chat window.
5. **Download:** Click **📥 DOWNLOAD HISTORY** to save your conversation.

---

## 🏗️ Local Setup
If you want to run this project locally:

1. **Clone the repo:**
   ```bash
   git clone [[https://github.com/Zoha56/RAG-CHATBOT.git](https://github.com/Zoha56/RAG-CHATBOT.git)]
   cd RAG-CHATBOT
