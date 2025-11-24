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
# BACKEND API KEY
# --------------------------------------------------------
GROQ_API_KEY = "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"


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
        self.initialize_llm()

    def initialize_llm(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            api_key=GROQ_API_KEY
        )

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
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if len(content) < 50:
                            continue

                        rel_path = os.path.relpath(file_path, directory_path)
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": rel_path}
                        ))
                    except:
                        pass

        return documents

    def load_repository(self, repo_url):
        if self.repo_path and os.path.exists(self.repo_path):
            shutil.rmtree(self.repo_path)

        self.repo_path = tempfile.mkdtemp()

        Repo.clone_from(repo_url, self.repo_path)

        documents = self.get_source_files(self.repo_path)
        if not documents:
            return "No valid code files found."

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)

        if len(split_docs) > 5000:
            split_docs = split_docs[:5000]

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
        You are a Senior Developer. You must answer based on the provided code context.

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
# STREAMLIT APP
# --------------------------------------------------------

# Create RAG system once and store in session
if "rag" not in st.session_state:
    st.session_state.rag = HeavyDutyRAG()

rag = st.session_state.rag

st.title("Heavy Duty Repo Chat – Streamlit Version")
repo_url = st.text_input("GitHub Repository URL")

if st.button("Clone & Index Repository"):
    with st.spinner("Processing repository..."):
        result = rag.load_repository(repo_url)
    st.success(result)

query = st.text_area("Ask something about the repository")

if st.button("Send"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag.chat_response(query)
        st.write(answer)
