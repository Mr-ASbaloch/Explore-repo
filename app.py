import streamlit as st
import os
import shutil
import tempfile
import torch
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

# --------------------------------------------------------
# RAG Class
# --------------------------------------------------------

class HeavyDutyRAG:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.repo_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initialize_llm(self, api_key):
        if not api_key.strip():
            return "Error: API Key is empty."

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            api_key=api_key.strip()
        )
        return "API Key set successfully."

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
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]

            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in allowed_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if len(content) < 50:
                                continue
                            rel_path = os.path.relpath(file_path, directory_path)
                            documents.append(Document(
                                page_content=content,
                                metadata={"source": rel_path}
                            ))
                    except Exception:
                        pass
        
        return documents

    def load_repository(self, repo_url):
        if not self.llm:
            return "Error: Set API key first."

        if self.repo_path and os.path.exists(self.repo_path):
            shutil.rmtree(self.repo_path)

        self.repo_path = tempfile.mkdtemp()

        try:
            Repo.clone_from(repo_url, self.repo_path)
        except Exception as e:
            return f"Clone error: {str(e)}"

        documents = self.get_source_files(self.repo_path)
        if not documents:
            return "No valid code files found."

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=2000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)

        MAX_CHUNKS = 5000
        if len(split_docs) > MAX_CHUNKS:
            split_docs = split_docs[:MAX_CHUNKS]

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )

        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name="heavy_duty_index"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})

        return f"Indexed {len(documents)} files successfully."

    def chat_response(self, message):
        if not self.retriever:
            return "Please load a GitHub repo first."

        template = """
        You are an expert Senior Developer. Answer strictly based on the provided code context.
        If mentioning code, reference the file name.

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


# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------

rag = HeavyDutyRAG()

st.title("Heavy Duty Repo Chat – Streamlit Version")
st.write("Smart RAG system for large GitHub repositories.")

# Sidebar settings
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password")

if st.sidebar.button("Set API Key"):
    msg = rag.initialize_llm(api_key)
    st.sidebar.success(msg)

repo_url = st.sidebar.text_input("GitHub Repo URL")

if st.sidebar.button("Clone & Index Repository"):
    with st.spinner("Cloning and indexing repository..."):
        msg = rag.load_repository(repo_url)
    st.sidebar.success(msg)

# Chat interface
st.header("Chat with the Repository")
query = st.text_area("Ask a question about the code...")

if st.button("Send"):
    if not query.strip():
        st.warning("Write a question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag.chat_response(query)
        st.write(answer)
