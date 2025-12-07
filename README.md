# 📄 Chat with PDFs — Free & Fast RAG App

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Groq](https://img.shields.io/badge/API-Groq%20(Free)-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A lightning-fast, 100% free tool that lets you chat with your documents using **Llama 3** (via Groq) and local vector search.

## 🚀 Key Features

* **⚡ Instant Q&A:** Get immediate, accurate answers drawn directly from your file content.
* **🧠 Smart Context Search:** Automatically pinpoints the exact sections of your document needed to answer your query.
* **💸 100% Free Usage:** Powered by Groq's free tier (Llama 3 models) and local CPU embeddings.
* **🔒 Privacy-First:** Document embeddings are processed locally on your machine; only relevant text chunks are sent to the LLM.
* **📂 Multi-PDF Support:** Upload and search across multiple documents simultaneously.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM:** Llama 3 (via [Groq API](https://groq.com/))
* **Vector Store:** FAISS (Local CPU)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Framework:** LangChain

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Key:**
    * Create a `.env` file in the root folder.
    * Add your free Groq key:
        ```ini
        GROQ_API_KEY=gsk_your_key_here
        ```

## ▶️ Usage

1.  Run the application:
    ```bash
    streamlit run app.py
    ```
2.  The app will open in your browser.
3.  **Upload PDFs** in the sidebar and click **Process**.
4.  **Ask questions** in the main chat interface.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.