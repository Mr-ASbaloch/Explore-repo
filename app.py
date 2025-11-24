import streamlit as st
import os
import shutil
import tempfile
import zipfile
from typing import List, Tuple

import torch
from git import Repo

# LangChain & embeddings imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------
# CONFIG: put your API key here
# -----------------------
GROQ_API_KEY = "eplace with your Groq API key"  # <-- replace with your Groq API key

# -----------------------
# Utility functions
# -----------------------

def ensure_session_state_keys():
    """Ensure required session_state keys exist and are initialized."""
    if "llm" not in st.session_state:
        try:
            st.session_state.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                api_key=GROQ_API_KEY.strip(),
            )
        except Exception as e:
            # keep llm as None and show error later
            st.session_state.llm = None
            st.session_state._llm_error = str(e)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "repo_path" not in st.session_state:
        st.session_state.repo_path = None
    if "history" not in st.session_state:
        st.session_state.history: List[Tuple[str, str]] = []
    if "last_index_msg" not in st.session_state:
        st.session_state.last_index_msg = ""
    if "device" not in st.session_state:
        st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"

def clean_temp_repo():
    """Remove repo directory if present."""
    path = st.session_state.get("repo_path")
    if path and os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception:
            pass
    st.session_state.repo_path = None

def clear_index():
    """Clear vectorstore and retriever safely."""
    # Chroma objects sometimes have a persist directory; attempt to delete if present.
    vs = st.session_state.get("vectorstore")
    try:
        if vs and hasattr(vs, "persist_directory") and vs.persist_directory:
            try:
                shutil.rmtree(vs.persist_directory)
            except Exception:
                pass
    except Exception:
        pass

    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.last_index_msg = ""

def allowed_extensions():
    return {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', 
        '.c', '.h', '.html', '.css', '.md', '.json', '.go', '.rs'
    }

def ignored_dirs():
    return {
        'node_modules', 'venv', '.git', '.github', '__pycache__',
        'dist', 'build', 'target', 'bin', 'obj', '.idea', '.vscode'
    }

def get_source_files(directory_path: str) -> List[Document]:
    """
    Smart crawler that prunes directories and returns a list of LangChain Document
    objects for allowed file types.
    """
    documents: List[Document] = []
    allowed_exts = allowed_extensions()
    ignored = ignored_dirs()

    for root, dirs, files in os.walk(directory_path):
        # In-place prune
        dirs[:] = [d for d in dirs if d not in ignored and not d.startswith('.')]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in allowed_exts:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if not content or len(content) < 50:
                        continue
                    rel_path = os.path.relpath(file_path, directory_path)
                    documents.append(Document(page_content=content, metadata={"source": rel_path}))
                except Exception:
                    # skip unreadable files
                    continue
    return documents

# -----------------------
# Indexing / loading functions
# -----------------------

def index_repository_from_directory(directory_path: str, progress_callback=None) -> str:
    """
    Given a directory path, split documents, embed and build a Chroma vectorstore.
    progress_callback: function(percent: float, text: str) -> None (optional)
    """
    if progress_callback:
        progress_callback(0.05, "Scanning files...")
    documents = get_source_files(directory_path)
    if not documents:
        return "No valid code files found."

    if progress_callback:
        progress_callback(0.20, f"Splitting {len(documents)} files into chunks...")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)

    MAX_CHUNKS = 5000
    if len(split_docs) > MAX_CHUNKS:
        split_docs = split_docs[:MAX_CHUNKS]

    if progress_callback:
        progress_callback(0.45, f"Creating embeddings on device {st.session_state.device.upper()}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": st.session_state.device}
    )

    if progress_callback:
        progress_callback(0.70, "Building vector index (Chroma)...")
    # Create a temporary directory for chroma persistence to allow cleanup later
    persist_dir = tempfile.mkdtemp(prefix="chroma_")
    try:
        vs = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name="heavy_duty_index",
            persist_directory=persist_dir
        )
    except TypeError:
        # fallback if persist_directory is not accepted by this Chroma wrapper
        vs = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name="heavy_duty_index"
        )
        # attach persist dir for cleanup later
        try:
            vs.persist_directory = persist_dir
        except Exception:
            pass

    st.session_state.vectorstore = vs
    st.session_state.retriever = vs.as_retriever(search_kwargs={"k": 7})

    if progress_callback:
        progress_callback(1.0, f"Indexed {len(documents)} files ({len(split_docs)} chunks).")
    st.session_state.last_index_msg = f"Indexed {len(documents)} files ({len(split_docs)} chunks)."

    return st.session_state.last_index_msg

def clone_and_index_from_github(repo_url: str, progress_callback=None) -> str:
    # Remove previous repo if exists
    if st.session_state.get("repo_path"):
        clean_temp_repo()
    repo_dir = tempfile.mkdtemp(prefix="repo_")
    st.session_state.repo_path = repo_dir

    if progress_callback:
        progress_callback(0.02, "Cloning repository...")

    try:
        Repo.clone_from(repo_url, repo_dir)
    except Exception as e:
        clean_temp_repo()
        clear_index()
        return f"Clone error: {str(e)}"

    # index
    try:
        return index_repository_from_directory(repo_dir, progress_callback)
    except Exception as e:
        clean_temp_repo()
        clear_index()
        return f"Indexing error: {str(e)}"

def extract_zip_to_dir(zip_local_path: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix="repo_zip_")
    try:
        with zipfile.ZipFile(zip_local_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e
    return temp_dir

def upload_zip_and_index(uploaded_file, progress_callback=None) -> str:
    if not uploaded_file:
        return "No file provided."
    temp_file = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        repo_dir = extract_zip_to_dir(temp_file)
    except Exception as e:
        return f"Failed to extract ZIP: {str(e)}"

    # set repo_path to this dir so it can be cleared later
    st.session_state.repo_path = repo_dir

    try:
        return index_repository_from_directory(repo_dir, progress_callback)
    except Exception as e:
        clean_temp_repo()
        clear_index()
        return f"Indexing error: {str(e)}"
    finally:
        # optionally remove uploaded zip temp file
        try:
            os.remove(temp_file)
        except Exception:
            pass

# -----------------------
# Chat function
# -----------------------

def chat_with_repo(question: str) -> str:
    retriever = st.session_state.get("retriever")
    llm = st.session_state.get("llm")
    if not retriever:
        return "Please load a GitHub repo or upload a ZIP and index it first."
    if not llm:
        err = st.session_state.get("_llm_error", "LLM initialization failed.")
        return f"LLM not available: {err}"

    template = """You are an expert senior developer. Answer strictly based on the provided code context.
Reference file names when relevant.

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        # chain.invoke may return a string or some other object; convert to str
        out = chain.invoke(question)
        return str(out)
    except Exception as e:
        return f"Chat error: {str(e)}"

# -----------------------
# Streamlit UI (Sidebar layout)
# -----------------------

st.set_page_config(page_title="ChatMyRepo", layout="wide")

ensure_session_state_keys()

# Sidebar
with st.sidebar:
    st.header("Repository / Index Controls")
    input_method = st.radio("Load repository from:", ("GitHub URL", "Upload ZIP"))

    status_text = st.empty()  # area for status messages
    progress_bar = st.progress(0.0)

    if input_method == "GitHub URL":
        repo_input = st.text_input("GitHub repo URL (https://...)")
        if st.button("Clone & Index"):
            status_text.info("Starting clone & index...")
            progress_bar.progress(0.0)
            msg = clone_and_index_from_github(repo_input, lambda p, t: (progress_bar.progress(min(max(p, 0.0), 1.0)), status_text.info(t)))
            status_text.success(msg)
    else:  # Upload ZIP
        uploaded = st.file_uploader("Upload a project ZIP file", type=["zip"])
        if st.button("Upload & Index"):
            if uploaded is None:
                st.warning("Please upload a ZIP file first.")
            else:
                status_text.info("Starting extraction & index...")
                progress_bar.progress(0.0)
                msg = upload_zip_and_index(uploaded, lambda p, t: (progress_bar.progress(min(max(p, 0.0), 1.0)), status_text.info(t)))
                status_text.success(msg)

    st.markdown("---")
    st.button("Clear chat history", key="clear_history_btn", on_click=lambda: st.session_state.history.clear())
    st.button("Clear repo & index", key="clear_repo_btn", on_click=lambda: (clean_temp_repo(), clear_index(), status_text.info("Repo and index cleared.")))

    st.markdown("**Index status:**")
    st.write(st.session_state.get("last_index_msg", "No index yet."))

    st.markdown("---")
    st.caption("Backend LLM status:")
    llm_ok = st.session_state.get("llm") is not None
    if llm_ok:
        st.write("LLM initialized.")
    else:
        st.write("LLM not initialized. Check GROQ_API_KEY and logs.")
        if "_llm_error" in st.session_state:
            st.text(st.session_state._llm_error)

# Main area: Chat UI
st.title("Heavy Duty -> SniffCode")
st.markdown("Ask questions about the code you indexed. The system will answer based on repository context only.")

col_main, col_side = st.columns([3, 1])

with col_main:
    user_input = st.text_area("Your question", height=140, key="user_question")
    if st.button("Send"):
        q = user_input.strip()
        if not q:
            st.warning("Please type a question.")
        else:
            with st.spinner("Querying..."):
                answer = chat_with_repo(q)
            # Append to history
            st.session_state.history.append(("You", q))
            st.session_state.history.append(("Assistant", answer))

    st.markdown("### Chat transcript")
    if not st.session_state.history:
        st.info("No messages yet. Load & index a repo, then ask questions.")
    else:
        for role, text in st.session_state.history:
            if role == "You":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")

with col_side:
    st.markdown("### Controls")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.success("Chat cleared.")
    if st.button("Show index info"):
        st.write("Repo path:", st.session_state.get("repo_path"))
        st.write("Vectorstore:", bool(st.session_state.get("vectorstore")))
        st.write("Retriever:", bool(st.session_state.get("retriever")))
        st.write("Device:", st.session_state.get("device"))

# Footer instructions
st.markdown("---")
st.caption("Replace the GROQ_API_KEY at the top of this file with your API key. If you deploy on Streamlit Cloud, set environment variables instead of hardcoding secrets in production.")
