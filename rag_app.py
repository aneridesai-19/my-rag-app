# rag_app.py

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

# Load OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not set in .env file.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

CHATS_FILE = "chat_sessions.json"
DOCS_FILE = "saved_docs.json"

# ---------- Utils ----------
def load_chats():
    return json.load(open(CHATS_FILE)) if os.path.exists(CHATS_FILE) else {}

def save_chats(chats):
    json.dump(chats, open(CHATS_FILE, "w"))

def load_docs():
    return json.load(open(DOCS_FILE)) if os.path.exists(DOCS_FILE) else {}

def save_docs(docs):
    json.dump(docs, open(DOCS_FILE, "w"))

def extract_text_from_pdf(pdf_file):
    return "\n".join([page.extract_text() or "" for page in PdfReader(pdf_file).pages])

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def build_vectorstore(all_chunks):
    vectorizer = TfidfVectorizer().fit(all_chunks)
    vectors = vectorizer.transform(all_chunks).toarray().astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors, vectorizer

def search_index(query, chunks, index, vectors, vectorizer, k=3):
    qv = vectorizer.transform([query]).toarray().astype('float32')
    D, I = index.search(qv, k)
    return [chunks[i] for i in I[0]]

def get_openai_answer(query, context, chat_history):
    system_prompt = f"""You are a helpful assistant. Use only the information from the context below to answer user questions.
If the answer is not directly stated, you may infer it **only if it's strongly implied**.

If multiple documents are used, synthesize the answer across them.
If a question is unclear, politely ask for clarification.

Context:
{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history
    messages.append({"role": "user", "content": query})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content.strip()

# ---------- Session Setup ----------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "docs" not in st.session_state:
    st.session_state.docs = load_docs()

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.chats[new_id] = {"name": "New Chat", "messages": []}

if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = False

# ---------- Vector Store Setup (Multi-doc Support) ----------
if st.session_state.docs:
    combined_chunks = []
    for doc in st.session_state.docs.values():
        combined_chunks.extend(split_text(doc["text"]))
    index, vectors, vectorizer = build_vectorstore(combined_chunks)
    st.session_state.index = index
    st.session_state.vectors = vectors
    st.session_state.vectorizer = vectorizer
    st.session_state.chunks = combined_chunks
else:
    st.session_state.index = None
    st.session_state.vectors = None
    st.session_state.vectorizer = None
    st.session_state.chunks = []

# ---------- Sidebar ----------
st.markdown("""
<style>
input[type="text"] { max-width: 180px !important; }
div.stButton > button { padding: 0.25em 0.6em; font-size: 0.85rem; margin-left: 4px; }
.active-chat > button { background-color: #2f2f2f !important; color: white !important; }
.active-doc > button { background-color: #2f2f2f !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📁 RAG Chatbot")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files and not st.session_state.uploaded_files_processed:
        for file in uploaded_files:
            if file.name in [doc["name"] for doc in st.session_state.docs.values()]:
                st.warning(f"⚠️ File '{file.name}' already uploaded, skipping.")
                continue

            file_id = str(uuid.uuid4())
            text = extract_text_from_pdf(file)
            st.session_state.docs[file_id] = {"name": file.name, "text": text}

        save_docs(st.session_state.docs)
        st.success("✅ Files uploaded!")
        st.session_state.uploaded_files_processed = True
        st.rerun()

    if not uploaded_files:
        st.session_state.uploaded_files_processed = False

    if st.session_state.docs:
        st.markdown("### 📄 Saved Documents")
        for doc_id, doc in st.session_state.docs.items():
            st.markdown(f"- {doc['name']}")

    st.markdown("### 💬 Chats")

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_id
        st.session_state.chats[new_id] = {"name": f"Chat {len(st.session_state.chats)+1}", "messages": []}
        save_chats(st.session_state.chats)
        st.rerun()
        
    for cid, chat in list(st.session_state.chats.items())[::-1]:
        is_active = (cid == st.session_state.current_chat_id)
        col1, col2 = st.columns([6, 1])
        with col1:
            if is_active:
                new_name = st.text_input("", value=chat["name"], key=f"chat-rename-{cid}", label_visibility="collapsed")
                if new_name != chat["name"]:
                    chat["name"] = new_name
                    save_chats(st.session_state.chats)
            else:
                if st.button(chat["name"], key=f"chat-open-{cid}", use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.rerun()
        with col2:
            if st.button("🗑️", key=f"del-chat-{cid}"):
                del st.session_state.chats[cid]

                # If deleted chat was the current one
                if st.session_state.current_chat_id == cid:
                    remaining_chats = list(st.session_state.chats.keys())
                    if remaining_chats:
                        st.session_state.current_chat_id = remaining_chats[0]  # pick another
                    else:
                        # Create a new default chat
                        new_id = str(uuid.uuid4())
                        st.session_state.current_chat_id = new_id
                        st.session_state.chats[new_id] = {"name": "New Chat", "messages": []}

                save_chats(st.session_state.chats)
                st.rerun()


    

# ---------- Main Chat Area ----------
st.title("🤖 Ask Your Documents")

if st.session_state.index and st.session_state.vectorizer and st.session_state.chunks:
    chat = st.session_state.chats[st.session_state.current_chat_id]
    for msg in chat["messages"]:
        st.markdown(f"👤 **You:** {msg['content']}" if msg["role"] == "user" else f"🤖 **Bot:**\n{msg['content']}", unsafe_allow_html=True)

    user_input = st.chat_input("Type your question...")

    if user_input:
        chat["messages"].append({"role": "user", "content": user_input})
        docs = search_index(user_input, st.session_state.chunks, st.session_state.index, st.session_state.vectors, st.session_state.vectorizer)
        context = "\n\n".join(docs)
        answer = get_openai_answer(user_input, context, chat["messages"])
        chat["messages"].append({"role": "assistant", "content": answer})
        st.session_state.chats[st.session_state.current_chat_id] = chat
        save_chats(st.session_state.chats)
        st.rerun()
else:
    st.info("Please upload at least one PDF document.")
