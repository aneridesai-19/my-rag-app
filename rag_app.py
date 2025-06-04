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

CHATS_FILE = "chat_sessions.json"
DOCS_FILE = "saved_docs.json"
MAX_CHAT_NAME_LEN = 20

# ------------------ UTILS ------------------

def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(CHATS_FILE, "w") as f:
        json.dump(chats, f)

def load_docs():
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_docs(docs):
    with open(DOCS_FILE, "w") as f:
        json.dump(docs, f)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def build_vectorstore(text_chunks):
    global vectorizer
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
    reply = response.choices[0].message.content.strip()
    return reply

# ------------------ SESSION SETUP ------------------

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "docs" not in st.session_state:
    st.session_state.docs = load_docs()

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.chats[new_id] = {"name": "New Chat", "messages": []}

if "current_doc_id" not in st.session_state or st.session_state.current_doc_id not in st.session_state.docs:
    if st.session_state.docs:
        last_doc_id = list(st.session_state.docs.keys())[-1]
        st.session_state.current_doc_id = last_doc_id
        chunks = split_text(st.session_state.docs[last_doc_id]["text"])
        index, vectors, vectorizer = build_vectorstore(chunks)
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.vectors = vectors
        st.session_state.vectorizer = vectorizer
    else:
        st.session_state.current_doc_id = None

if "index" not in st.session_state:
    st.session_state.index = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ------------------ SIDEBAR CSS & UI ------------------

st.markdown(
    """
    <style>
    input[type="text"] {
        max-width: 180px !important;
    }
    div.stButton > button {
        padding: 0.25em 0.6em;
        font-size: 0.85rem;
        margin-left: 4px;
    }
    .active-chat > button {
        background-color: #000 !important;
        color: white !important;
    }
    .active-doc > button {
        background-color: #000 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📁 RAG Chatbot")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_id = str(uuid.uuid4())
            text = extract_text_from_pdf(file)
            st.session_state.docs[file_id] = {
                "name": file.name,
                "text": text
            }
            st.session_state.current_doc_id = file_id  # Auto select last uploaded

        save_docs(st.session_state.docs)
        st.success("✅ Files uploaded and saved!")

        chunks = split_text(text)
        index, vectors, vectorizer = build_vectorstore(chunks)
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.vectors = vectors
        st.session_state.vectorizer = vectorizer

        st.markdown("---")

    st.markdown("### 📄 Saved Documents")
    for doc_id, doc in list(st.session_state.docs.items())[::-1]:
        is_selected = (doc_id == st.session_state.current_doc_id)
        css_class = "active-doc" if is_selected else ""

        with st.container():
            col1, col2 = st.columns([6, 1])
            with col1:
                if st.button(doc["name"], key=f"doc-{doc_id}", use_container_width=True):
                    st.session_state.current_doc_id = doc_id
                    chunks = split_text(doc["text"])
                    index, vectors, vectorizer = build_vectorstore(chunks)
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.vectors = vectors
                    st.session_state.vectorizer = vectorizer
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del-doc-{doc_id}"):
                    st.session_state.docs.pop(doc_id, None)
                    save_docs(st.session_state.docs)
                    if st.session_state.current_doc_id == doc_id:
                        st.session_state.current_doc_id = None
                        st.session_state.chunks = []
                        st.session_state.index = None
                        st.session_state.vectors = None
                        st.session_state.vectorizer = None
                    st.rerun()

    st.markdown("---")

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_id
        st.session_state.chats[new_id] = {"name": f"Chat {len(st.session_state.chats) + 1}", "messages": []}
        save_chats(st.session_state.chats)
        st.rerun()

    st.markdown("### 💬 Chats")
    for cid, chat in list(st.session_state.chats.items())[::-1]:
        is_active = (cid == st.session_state.current_chat_id)
        css_class = "active-chat" if is_active else ""
        col1, col2 = st.columns([6, 1])
        with col1:
            if is_active:
                new_name = st.text_input(
                    label="",
                    value=chat["name"],
                    key=f"chat-rename-{cid}",
                    label_visibility="collapsed"
                )
                if new_name != chat["name"]:
                    st.session_state.chats[cid]["name"] = new_name
                    save_chats(st.session_state.chats)
            else:
                if st.button(chat["name"], key=f"chat-open-{cid}", use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.rerun()
        with col2:
            if st.button("🗑️", key=f"del-chat-{cid}"):
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

# ------------------ MAIN UI ------------------

st.title("🤖 Ask Your Documents")

if st.session_state.current_doc_id:
    chat_history = st.session_state.chats.get(st.session_state.current_chat_id, {}).get("messages", [])

    for msg in chat_history:
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(f"👤 **You:** {content}")
        elif role == "assistant":
            st.markdown(f"🤖 **Bot:**\n{content}", unsafe_allow_html=True)

    user_input = st.chat_input("Type your question...")

    if user_input:
        if st.session_state.current_chat_id is None:
            new_id = str(uuid.uuid4())
            st.session_state.current_chat_id = new_id
            st.session_state.chats[new_id] = {"name": f"Chat {len(st.session_state.chats)+1}", "messages": []}
            save_chats(st.session_state.chats)

        chat = st.session_state.chats[st.session_state.current_chat_id]
        chat["messages"].append({"role": "user", "content": user_input})

        if st.session_state.index and st.session_state.vectorizer and st.session_state.chunks:
            docs = search_index(user_input, st.session_state.chunks, st.session_state.index, st.session_state.vectors, st.session_state.vectorizer)
            context = "\n\n".join(docs)
            answer = get_openai_answer(user_input, context, chat["messages"])
        else:
            answer = "Please upload a document first."

        chat["messages"].append({"role": "assistant", "content": answer})
        st.session_state.chats[st.session_state.current_chat_id] = chat
        save_chats(st.session_state.chats)
        st.rerun()
else:
    st.info("Please upload a PDF document to begin.")
