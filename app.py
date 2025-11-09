import streamlit as st
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from docx import Document
import fitz  # PyMuPDF
import speech_recognition as sr

# --- Basic Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API Key not found. Please set it in the .env file.")
    st.stop()

# --- Helper: Safe text extraction ---
def safe_extract_text(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
    except Exception:
        pass
    return "(No content returned — possibly cold start or quota limit)"

# --- Helper: Robust Gemini call ---
def robust_generate(prompt, retries=2):
    models_to_try = ["gemini-2.0-flash-exp", "gemini-1.5-flash"]
    for model_name in models_to_try:
        model = genai.GenerativeModel(model_name)
        for attempt in range(retries + 1):
            try:
                resp = model.generate_content(prompt)
                text = safe_extract_text(resp)
                if text.strip() and "No content returned" not in text:
                    return text
            except Exception as e:
                if "429" in str(e):
                    break
                if attempt < retries:
                    time.sleep(1)
                    continue
                return f"(Error: {e})"
    return "(No model could return a valid response)"

# --- Warm-up ---
try:
    robust_generate("Hello")
except Exception as e:
    print(f"Warm-up failed: {e}")

# --- Document Processing ---
def get_document_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.name.endswith('.pdf'):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
        elif file.name.endswith('.docx'):
            document = Document(file)
            for para in document.paragraphs:
                text += para.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("Could not extract text from documents. Please check the files.")
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.raw_text = "\n".join(text_chunks)
        st.sidebar.success("Documents Processed Successfully!", icon="✅")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()

def get_conversational_chain():
    prompt_template = """You are LegalAI, an intelligent legal assistant designed to help users understand legal documents, 
    contracts, and legal concepts. Answer questions accurately based on the provided context. If you're not sure, say so.
    
    Context:\n {context}\n 
    Question: \n{question}\n 
    Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_summary_chain():
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2, google_api_key=api_key)
    return load_summarize_chain(model, chain_type="map_reduce")

# --- Speech to Text ---
def record_audio():
    """Record audio from microphone and convert to text"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = r.listen(source, timeout=5, phrase_time_limit=10)
            st.success("Recording complete! Processing...")
            
            # Recognize speech using Google
            text = r.recognize_google(audio_data)
            return text
        except sr.WaitTimeoutError:
            st.error("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.error("Could not understand audio. Please speak clearly and try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from speech recognition service: {e}")
            return None
        except Exception as e:
            st.error(f"Error: {e}")
            return None

# --- QnA & Summarization ---
def handle_document_qna(user_question):
    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def handle_general_qna(user_question):
    prompt = f"You are LegalAI, a helpful Legal AI Assistant. Answer the following legal question: {user_question}"
    return robust_generate(prompt)

def generate_summary(custom_instruction):
    if "raw_text" not in st.session_state or not st.session_state.raw_text:
        st.error("No document text found to summarize.")
        return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([st.session_state.raw_text])
    chain = get_summary_chain()
    with st.spinner("Generating detailed summary..."):
        initial_summary = chain.run(docs)
        refine_prompt = f"Given the following summary, refine it to follow: '{custom_instruction}'.\n\nSUMMARY:\n{initial_summary}"
        final_summary = robust_generate(refine_prompt)
    st.session_state.chat_history.append(("LegalAI", f"**Summary (Instruction: *{custom_instruction}*)**\n\n" + final_summary))

def translate_text(text, target_language):
    prompt = f"Translate the following English text to {target_language}. Only provide the translation:\n\n{text}"
    return robust_generate(prompt)

# --- Page Config ---
st.set_page_config(
    page_title="LegalAI - Your Legal Assistant", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS with Light Blue & White Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styling */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(14, 165, 233, 0.1);
        margin-bottom: 2rem;
        text-align: center;
        border: 2px solid #bae6fd;
    }
    
    .main-header h1 {
        color: #0284c7;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        color: #64748b;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 2px solid #e0f2fe;
        box-shadow: 4px 0 15px rgba(14, 165, 233, 0.08);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.08);
        border: 1px solid #e0f2fe;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0ea5e9;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: white;
        border-left: 4px solid #06b6d4;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5);
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        border-radius: 12px;
        border: 2px solid #bae6fd;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #0ea5e9;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px dashed #bae6fd;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0ea5e9;
        background: #f0f9ff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.08);
        border: 1px solid #e0f2fe;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: #d1fae5;
        color: #065f46;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #10b981;
    }
    
    .stError {
        background: #fee2e2;
        color: #991b1b;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #ef4444;
    }
    
    .stInfo {
        background: #dbeafe;
        color: #1e40af;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        font-weight: 600;
        color: #0284c7;
        border: 1px solid #e0f2fe;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        border-top: 2px solid #e0f2fe;
        padding-top: 1rem;
        background: white;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #0ea5e9 !important;
    }
    
    /* Sidebar Headers */
    .sidebar .element-container h1,
    .sidebar .element-container h2,
    .sidebar .element-container h3 {
        color: #0284c7;
        font-weight: 700;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #bae6fd, transparent);
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

# --- Main Header ---
st.markdown("""
    <div class="main-header">
        <h1>⚖️ LegalAI</h1>
        <p>Your Intelligent Legal Assistant</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📂 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload Legal Documents", 
        type=["pdf", "docx"], 
        accept_multiple_files=True,
        help="Upload PDF or DOCX files to analyze"
    )
    
    if st.button("🔍 Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Analyzing documents..."):
                raw_text = get_document_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
        else:
            st.warning("Please upload at least one document.", icon="⚠️")
    
    st.markdown("---")
    
    if st.session_state.vector_store:
        st.markdown("## 📝 Generate Summary")
        custom_instruction = st.text_input(
            "Custom Instruction:",
            "Provide a comprehensive summary highlighting key legal points.",
            help="Customize how you want the summary"
        )
        if st.button("✨ Generate Summary", use_container_width=True):
            generate_summary(custom_instruction)
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("## 📜 Conversation History")
    with st.expander("View History", expanded=False):
        if st.session_state.chat_history:
            for role, text in st.session_state.chat_history[-5:]:
                if role == "You":
                    st.text(f"👤 You: {text[:50]}...")
                else:
                    st.text(f"⚖️ AI: {text[:50]}...")
        else:
            st.text("No questions asked yet.")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.raw_text = ""
        st.rerun()

# --- Main Content Tabs ---
chat_tab, translate_tab = st.tabs(["💬 Chat", "🌐 Translate"])

# --- Chat Tab ---
with chat_tab:
    st.markdown("### Ask Questions or Get Legal Assistance")
    
    # Display chat history
    for role, text in st.session_state.chat_history:
        with st.chat_message(role, avatar="👤" if role == "You" else "⚖️"):
            st.markdown(text)
    
    # Input section with voice button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        prompt = st.chat_input("Ask a question about your document or a general legal query...")
    
    with col2:
        if st.button("🎤 Voice", use_container_width=True, help="Click to use voice input"):
            voice_text = record_audio()
            if voice_text:
                st.success(f"Recognized: {voice_text}")
                prompt = voice_text
    
    # Handle user input
    if prompt:
        st.session_state.chat_history.append(("You", prompt))
        
        with st.chat_message("You", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("LegalAI", avatar="⚖️"):
            with st.spinner("Thinking..."):
                if st.session_state.vector_store:
                    response_text = handle_document_qna(prompt)
                else:
                    response_text = handle_general_qna(prompt)
                st.markdown(response_text)
        
        st.session_state.chat_history.append(("LegalAI", response_text))
        st.rerun()

# --- Translate Tab ---
with translate_tab:
    st.markdown("### Legal Document Translation")
    
    text_to_translate = st.text_area(
        "Enter English text to translate:", 
        height=200,
        placeholder="Type or paste legal text here..."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_language = st.selectbox(
            "Select Target Language:", 
            ["Hindi (हिंदी)", "Marathi (मराठी)"]
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        translate_button = st.button("🔄 Translate", use_container_width=True)
    
    if translate_button and text_to_translate:
        language = target_language.split()[0]  # Extract language name
        with st.spinner(f"Translating to {language}..."):
            translated_text = translate_text(text_to_translate, language)
            if translated_text:
                st.success("Translation Complete!")
                st.markdown(f"### Translated Text ({language}):")
                st.info(translated_text)
    elif translate_button:
        st.warning("Please enter text to translate.")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #0284c7; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(14, 165, 233, 0.1);'>
        <p style='margin: 0;'>⚖️ <b>LegalAI</b> - Powered by Google Gemini AI</p>
        <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'>Your trusted legal assistant for document analysis and legal queries</p>
    </div>
""", unsafe_allow_html=True)
