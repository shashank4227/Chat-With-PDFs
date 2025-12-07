import os
import pathlib
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- 1. FREE LOCAL EMBEDDINGS ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- 2. VECTOR STORE ---
from langchain_community.vectorstores import FAISS

# --- 3. FREE LLM (Groq) ---
from langchain_groq import ChatGroq

# --- TEXT SPLITTER ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------
load_dotenv()

# -------------------------------------------------------
# Setup Embedding Model (Local CPU)
# -------------------------------------------------------
@st.cache_resource
def get_embed_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embed_model = get_embed_model()

# -------------------------------------------------------
# Directories
# -------------------------------------------------------
FAISS_DIR = pathlib.Path("faiss_index")
FAISS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_text(text)

def build_faiss(chunks):
    index = FAISS.from_texts(
        texts=chunks,
        embedding=embed_model, 
        metadatas=[{"source": "pdf"}] * len(chunks)
    )
    index.save_local(str(FAISS_DIR))
    st.success("✅ Index Created Successfully!")

def load_faiss():
    if not (FAISS_DIR / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings=embed_model,
        allow_dangerous_deserialization=True
    )

def generate_answer_groq(api_key, context, question):
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    
    prompt = f"""
    Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    
    response = llm.invoke(prompt)
    return response.content

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot (Auto-Auth)", layout="wide")
st.title("Chat with PDFs")

# --- SIDEBAR ---
with st.sidebar:
    # st.header("🔑 Setup")
    
    # Check if Key exists in .env
    env_key = os.getenv("GROQ_API_KEY")
    
    if env_key:
        # st.success("✅ API Key loaded from .env")
        groq_api_key = env_key
    else:
        # Fallback: Ask user to input if not in .env
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")
        if not groq_api_key:
            st.warning("⚠️ Key missing. Add GROQ_API_KEY to .env or enter here.")
            st.markdown("[Get Free Key](https://console.groq.com/keys)")

    # st.divider()

    st.header("📄 Upload")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if not uploaded_pdfs:
            st.warning("Please upload PDFs first.")
        else:
            with st.spinner("Indexing (Local CPU)..."):
                text = extract_text_from_pdfs(uploaded_pdfs)
                chunks = split_text(text)
                build_faiss(chunks)

# --- MAIN CHAT ---
st.header("Ask Questions")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not groq_api_key:
        st.error("Missing API Key. Please check your settings.")
    else:
        db = load_faiss()
        if not db:
            st.error("No index found. Please process PDFs first.")
        else:
            # 1. Search
            with st.spinner("Searching docs..."):
                results = db.similarity_search(question, k=4)
                context = "\n\n".join([doc.page_content for doc in results])
            
            # 2. Answer
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer_groq(groq_api_key, context, question)
                    st.subheader("Answer:")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error: {e}")