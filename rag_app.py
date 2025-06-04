import streamlit as st
from PyPDF2 import PdfReader
import os
import faiss
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from dotenv import load_dotenv
import uuid
import json

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Load OpenAI API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not set in .env file.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

CHAT_SESSIONS_FILE = "chat_sessions.json"

# ------------------ UTILS ------------------

def load_chats():
    if os.path.exists(CHAT_SESSIONS_FILE):
        with open(CHAT_SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(CHAT_SESSIONS_FILE, "w") as f:
        json.dump(chats, f)

def save_uploaded_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        docs.append(text)
    return "\n".join(docs)

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append("".join(words[i:i + chunk_size]))
    return chunks

def build_vectorstore(text_chunks):
    vectorizer = TfidfVectorizer().fit(text_chunks)
    vectors = vectorizer.transform(text_chunks).toarray().astype('float32')
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors, vectorizer

def search_index(query, chunks, index, vectors, vectorizer, k=3):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

def get_openai_answer(query, context, chat_history):
    system_prompt = f"""You are a helpful assistant.
Only answer using the information provided in the context below.
If the answer is not explicitly mentioned, respond with: "I couldn't find that information in the document."

Context:
{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# ------------------ SESSION SETUP ------------------

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.chats[new_id] = {"name": "New Chat", "messages": []}

if "index" not in st.session_state:
    st.session_state.index = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ------------------ SIDEBAR CSS & UI ------------------

st.markdown("""
    <style>
    button[kind="secondary"] {
        background-color: #f0f2f6;
        color: #333;
        border-radius: 8px;
        padding: 0.5em 1em;
        text-align: left;
        width: 100%;
        transition: all 0.2s ease;
    }
    button[kind="secondary"]:hover {
        background-color: #dfe4ea;
        color: black;
        font-weight: 600;
    }
    input[type="text"] {
        max-width: 100%;
        padding: 4px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📁 RAG Chatbot")

    for cid, chat in list(st.session_state.chats.items())[::-1]:
        with st.container():
            cols = st.columns([8, 1])
            with cols[0]:
                if st.button(f"💬 {chat['name']}", key=f"chat-btn-{cid}"):
                    st.session_state.current_chat_id = cid
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"del-{cid}"):
                    del st.session_state.chats[cid]
                    if cid == st.session_state.current_chat_id:
                        if st.session_state.chats:
                            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                        else:
                            new_id = str(uuid.uuid4())
                            st.session_state.current_chat_id = new_id
                            st.session_state.chats[new_id] = {"name": "New Chat", "messages": []}
                    save_chats(st.session_state.chats)
                    st.rerun()

            new_name = st.text_input("✏️ Rename", value=chat["name"], key=f"rename-{cid}")
            if new_name != chat["name"]:
                st.session_state.chats[cid]["name"] = new_name
                save_chats(st.session_state.chats)

        st.markdown("---")

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_id
        st.session_state.chats[new_id] = {"name": f"Chat {len(st.session_state.chats)+1}", "messages": []}
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.vectors = None
        st.session_state.vectorizer = None
        save_chats(st.session_state.chats)
        st.rerun()

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        text = save_uploaded_files(uploaded_files)
        chunks = split_text(text)
        index, vectors, vectorizer = build_vectorstore(chunks)
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.vectors = vectors
        st.session_state.vectorizer = vectorizer
        st.success("✅ Files processed!")

# ------------------ MAIN CHAT UI ------------------

st.title("🤖 Ask Your Documents")

chat = st.session_state.chats[st.session_state.current_chat_id]

for msg in chat["messages"]:
    role, content = msg["role"], msg["content"]
    if role == "user":
        st.markdown(f"👤 **You:** {content}")
    elif role == "assistant":
        st.markdown(f"🤖 **Bot:**\n{content}", unsafe_allow_html=True)

user_input = st.chat_input("Type your question...")

if user_input:
    chat["messages"].append({"role": "user", "content": user_input})
    messages = chat["messages"]

    if st.session_state.index and st.session_state.vectorizer and st.session_state.chunks:
        docs = search_index(user_input, st.session_state.chunks, st.session_state.index, st.session_state.vectors, st.session_state.vectorizer)
        context = "\n\n".join(docs)
        answer = get_openai_answer(user_input, context, messages)
    else:
        answer = "Please upload a document first."

    chat["messages"].append({"role": "assistant", "content": answer})
    st.session_state.chats[st.session_state.current_chat_id] = chat
    save_chats(st.session_state.chats)
    st.rerun()
