import streamlit as st
import os
import shutil
import tempfile
import torch
import zipfile
from git import Repo

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ----------------------------
# BACKEND API KEY
# ----------------------------
GROQ_API_KEY = "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"


# ----------------------------
# RAG CLASS
# ----------------------------
class HeavyDutyRAG:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.repo_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialize_llm()

    def initialize_llm(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            api_key=GROQ_API_KEY
        )

    def extract_uploaded_zip(self, uploaded_zip):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir

    def get_source_files(self, directory_path):
        documents = []
        allowed_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
            '.html', '.css', '.md', '.json', '.go', '.rs'
        }
        ignored_dirs = {
            'node_modules', 'venv', '.git', '.github', '__pycache__',
            'dist', 'build', 'target', 'bin', 'obj', '.idea', '.vscode'
        }

        for root, dirs, files in os.walk(directory_path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in allowed_extensions:
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if len(content) < 50:
                            continue
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": os.path.relpath(file_path, directory_path)}
                        ))
                    except:
                        pass
        return documents

    def index_repository(self, directory_path, progress):
        progress.progress(0.2, "Reading source files...")
        documents = self.get_source_files(directory_path)
        if not documents:
            return "No valid code files found."

        progress.progress(0.4, "Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)

        if len(split_docs) > 5000:
            split_docs = split_docs[:5000]

        progress.progress(0.6, "Embedding chunks...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )

        progress.progress(0.8, "Building vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name="heavy_duty_index",
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})

        progress.progress(1.0, "Repository indexed successfully.")
        return f"Indexed {len(documents)} files."

    def load_repository_from_url(self, repo_url, progress):
        if self.repo_path and os.path.exists(self.repo_path):
            shutil.rmtree(self.repo_path)

        self.repo_path = tempfile.mkdtemp()

        progress.progress(0.1, "Cloning GitHub repository...")
        Repo.clone_from(repo_url, self.repo_path)

        return self.index_repository(self.repo_path, progress)

    def load_repository_from_zip(self, uploaded_zip, progress):
        if self.repo_path and os.path.exists(self.repo_path):
            shutil.rmtree(self.repo_path)

        progress.progress(0.1, "Extracting uploaded ZIP...")
        self.repo_path = self.extract_uploaded_zip(uploaded_zip)

        return self.index_repository(self.repo_path, progress)

    def chat_response(self, message):
        if not self.retriever:
            return "Please load a GitHub repo or upload a ZIP first."

        template = """
        You are a senior developer. Answer strictly from the code context.
        Reference file names when needed.

        Context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(message)


# ----------------------------
# STREAMLIT APP
# ----------------------------

# Keep RAG instance alive
if "rag" not in st.session_state:
    st.session_state.rag = HeavyDutyRAG()

rag = st.session_state.rag

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Heavy Duty Repo Chat – Advanced Version")

st.subheader("Load Repository")
option = st.radio("Choose input method:", ["GitHub URL", "Upload ZIP"])

progress = st.progress(0)

if option == "GitHub URL":
    repo_url = st.text_input("GitHub Repository URL")
    if st.button("Load Repository"):
        with st.spinner("Processing..."):
            msg = rag.load_repository_from_url(repo_url, progress)
        st.success(msg)

else:
    uploaded_zip = st.file_uploader("Upload a project ZIP file", type=["zip"])
    if st.button("Upload & Index"):
        if uploaded_zip:
            with st.spinner("Processing ZIP..."):
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_zip.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_zip.read())
                msg = rag.load_repository_from_zip(temp_path, progress)
            st.success(msg)
        else:
            st.warning("Please upload a ZIP file.")

st.subheader("Chat with Repository")

query = st.text_area("Your question")

col1, col2 = st.columns(2)

with col1:
    if st.button("Send"):
        if query.strip():
            answer = rag.chat_response(query)
            st.session_state.history.append(("You", query))
            st.session_state.history.append(("Assistant", answer))
        else:
            st.warning("Enter a question.")

with col2:
    if st.button("Clear Chat"):
        st.session_state.history = []


# Display chat history
st.write("### Chat History")
for role, message in st.session_state.history:
    st.write(f"**{role}:** {message}")
