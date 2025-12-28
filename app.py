import os
import json
import gradio as gr
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# --- Neon Custom Styling ---
custom_css = """
body { background-color: #050505; }
.gradio-container { border: 1px solid #00f2ff44 !important; border-radius: 15px !important; }

#header-text { 
    text-align: center; 
    text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff; 
    color: #00f2ff;
    margin-bottom: 20px;
}

.status-card { 
    background: rgba(15, 15, 25, 0.9) !important; 
    border: 1px solid #00f2ff !important; 
    padding: 15px; 
    border-radius: 12px; 
}

.neon-btn {
    background: linear-gradient(45deg, #00f2ff, #7000ff) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    box-shadow: 0 0 10px rgba(0, 242, 255, 0.4) !important;
    cursor: pointer;
}

#footer-text { text-align: center; font-size: 0.8em; color: #444; margin-top: 20px; }
"""

# --- Backend logic (Logic & Imports Preserved) ---
chain_holder = {"chain": None}

def build_kb(files, temperature=0.1):
    if not files:
        return "⚠️ Please upload files first."
    try:
        documents = []
        for f in files:
            loader = PyPDFLoader(f.name)
            documents.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True, 
            output_key='answer'
        )
        chain_holder["chain"] = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=vector_db.as_retriever(), 
            memory=memory, 
            return_source_documents=True
        )
        return "🟢 SUCCESS: Documents Indexed"
    except Exception as e:
        return f"🔴 ERROR: {str(e)}"

def predict_message(message, history):
    if chain_holder["chain"] is None:
        return "Please upload PDFs and initialize the system first."
    try:
        response = chain_holder["chain"].invoke({"question": message})
        answer = response.get('answer', "No response found.")
        sources = list(set([f"Page {d.metadata['page']+1}" for d in response.get('source_documents', [])]))
        if sources:
            answer += f"\n\n--- \n📍 **Verified Sources:** {', '.join(sources)}"
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"
#-------Download chat ------
def export_history(history):
    if not history:
        return None
    file_path = "chat_history.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        for msg in history:
            role = msg['role'].upper()
            content = msg['content']
            f.write(f"{role}: {content}\n\n")
    return file_path

# ---UI Layout ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="blue")) as demo:
    with gr.Row(elem_id="header-text"):
        gr.Markdown("# 🌌 RAG TERMINAL")
        gr.Markdown("Transforming static PDFs into interactive intelligence.")

    with gr.Row():
        # Sidebar
        with gr.Column(scale=1):
            with gr.Group(elem_classes="status-card"):
                gr.Markdown("### 📂 Data Injection")
                file_input = gr.File(label="PDF Files", file_count="multiple", file_types=[".pdf"])
                
                with gr.Accordion("🛠️ Advanced Tuning", open=False):
                    temp_slider = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.1, label="Temperature")
                
                process_btn = gr.Button("🚀 INITIALIZE SYSTEM", elem_classes="neon-btn")
            
            with gr.Group(elem_classes="status-card"):
                gr.Markdown("### 📡 System Status")
                status_box = gr.Markdown("🔴 **Offline** - Awaiting documents")

            with gr.Group(elem_classes="status-card"):
                gr.Markdown("### 💾 Session Data")
                download_btn = gr.DownloadButton("📥 DOWNLOAD HISTORY", elem_classes="neon-btn")

        # Chat
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=predict_message,
                type="messages",
                examples=["Summarize the document", "What are the main findings?"],
            )

    gr.Markdown("© 2025 Powered by Groq & LangChain Classic", elem_id="footer-text")

    # Connect components
    process_btn.click(
        fn=build_kb, 
        inputs=[file_input, temp_slider], 
        outputs=[status_box]
    )

    # Download interaction
    # Note: chatbot.chatbot refers to the internal state of the ChatInterface
    download_btn.click(
        fn=export_history,
        inputs=[chatbot.chatbot],
        outputs=[download_btn]
    )

if __name__ == "__main__":
    demo.launch()