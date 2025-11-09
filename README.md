# ⚖️ LegalAI  

**LegalAI** is a **Streamlit-based intelligent legal assistant** for analyzing, summarizing, and interacting with **legal documents**.  
It uses **Retrieval-Augmented Generation (RAG)** with **FAISS** vector search and **Google Gemini (via `google-generativeai`)** to deliver smart, context-aware responses — along with **speech input** and **multilingual translation** features.  

---

## 🧩 Overview  

This repository contains a single-page **Streamlit app (`app.py`)** that allows you to:  
- Upload **PDF/DOCX** legal files  
- Create **vector embeddings**  
- Ask **context-aware questions**  
- Generate **summaries and translations**  
- Interact using **voice commands**  

---

## 🚀 Key Features  

✅ **📄 Document Upload** – Upload and extract text from **PDF** or **DOCX** (via `PyMuPDF` / `python-docx`).  
✅ **🔍 Semantic Search** – Chunk text and store embeddings in a **FAISS** vector database.  
✅ **💬 Legal Q&A** – Ask document-grounded legal questions powered by **LangChain + Gemini**.  
✅ **🧾 Smart Summaries** – Generate and refine summaries using Gemini’s contextual understanding.  
✅ **🌐 Multilingual Translation** – Translate legal content into **Hindi** and **Marathi**.  
✅ **🎙️ Speech-to-Text** – Record voice queries with `speech_recognition` and get instant responses.  

---

## 🛠️ Tech Stack  

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend / AI** | LangChain, Google Gemini (via `google-generativeai`) |
| **Vector Store** | FAISS (`faiss-cpu`) |
| **Document Parsing** | PyMuPDF (`fitz`), `python-docx` |
| **Speech Input** | `speech_recognition` |
| **Environment Handling** | `python-dotenv` |

> 🐍 **Requires Python 3.8+**

---

## ⚡ Quick Start (Windows / PowerShell)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/NinadGawali/LegalAI.git
   cd LegalAI
   ```
2. **Create and activate a virtual environment**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file in the project root and add your Google Gemini API key:**

```bash
GEMINI_API_KEY=your_google_gemini_api_key_here
```

5. **Run the Streamlit app:**

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501) in your browser.

## 💡 How to Use  

🧾 **Step 1:** Upload one or more **PDF/DOCX** files from the sidebar.  
⚙️ **Step 2:** Click **"Process Documents"** to extract text, chunk it, and build a vector store.  
💬 **Step 3:** Ask questions in the chat box — responses are grounded in the uploaded docs.  
🪄 **Step 4:** Use **"Generate Summary"** to get concise or detailed summaries (customizable).  
🌐 **Step 5:** Translate legal text into **Hindi** or **Marathi** from the **Translate** tab.  
🎙️ **Step 6:** Record a voice query using the **microphone button** and send it directly.  

---

## ⚙️ Configuration  

- **API Key:** Add `GEMINI_API_KEY` in your `.env` file.  
- **Models:** Modify embedding and chat model settings inside `app.py` (LangChain adapters).  

---

## 🧰 Troubleshooting  

⚠️ **Google API Key Error:**  
Ensure `.env` exists in the root directory and contains `GEMINI_API_KEY`. Restart Streamlit.  

🎧 **Speech Recognition Issues:**  
If `pyaudio` fails to install on Windows, use pre-built wheels or disable audio features.  

💾 **FAISS Installation Problems:**  
Try using a **Conda environment** if `pip install faiss-cpu` fails.  

🔒 **Privacy Note:**  
Since this app calls the **Gemini API**, avoid uploading **confidential legal documents** unless you understand the implications.  

---

## 🧑‍💻 Development
- The app is a single module (`app.py`). To extend it consider:
	- Adding more robust error handling and logging
	- Extracting functionality into modules (parsing, embeddings, chain logic)
	- Adding tests for text extraction, chunking, and the conversational chain

## ⚖️ License
This project includes a `LICENSE` file. Please review it for licensing details. 
