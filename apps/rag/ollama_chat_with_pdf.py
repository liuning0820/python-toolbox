import os
import time

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from dotenv import load_dotenv

load_dotenv()

FILES_DIR = "files"
VECTOR_DIR = "vectors"

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:32b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")

# override with environment variables if set
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_BASE_URL = os.getenv(
    "OLLAMA_EMBEDDING_BASE_URL", "http://localhost:11434"
)


def ensure_dirs():
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)


def init_session_state():
    if "template" not in st.session_state:
        st.session_state.template = (
            "You are a knowledgeable chatbot, here to help with questions of the user. "
            "Your tone should be professional and informative.\n\n"
            "Context: {context}\nHistory: {history}\n\nUser: {question}\nChatbot:"
        )
    if "prompt" not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template,
        )
    if "memory" not in st.session_state:
        memory = ConversationBufferMemory(
            memory_key="history", return_messages=True, input_key="question"
        )

        st.session_state.memory = memory
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=OllamaEmbeddings(
                base_url=OLLAMA_EMBEDDING_BASE_URL, model=OLLAMA_EMBEDDING_MODEL
            ),
        )
    if "llm" not in st.session_state:
        st.session_state.llm = OllamaLLM(
            base_url=OLLAMA_LLM_BASE_URL,
            model=OLLAMA_LLM_MODEL,
            verbose=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def save_pdf(uploaded_file):
    filename = uploaded_file.name
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    file_path = os.path.join(FILES_DIR, filename)
    if not os.path.isfile(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    return file_path


def process_pdf(file_path):
    """
    增量式向量化PDF内容，避免重复向量化。
    """
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, length_function=len
    )
    all_splits = text_splitter.split_documents(data)
    # 加载现有向量库或新建
    vectorstore = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=OllamaEmbeddings(
            base_url=OLLAMA_EMBEDDING_BASE_URL, model=OLLAMA_EMBEDDING_MODEL
        ),
    )
    # 修正：遍历 metadatas 判断是否已向量化该文件
    metadatas = vectorstore.get().get("metadatas", [])
    sources = [meta.get("source", "") for meta in metadatas if isinstance(meta, dict)]
    if file_path not in sources:
        vectorstore.add_documents(all_splits)

    st.session_state.vectorstore = vectorstore


def get_qa_chain():
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(),
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            },
        )
    return st.session_state.qa_chain


def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, message_placeholder):
        self.message_placeholder = message_placeholder
        self.tokens = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += token
        self.message_placeholder.markdown(self.tokens + "▌")


def main():
    ensure_dirs()
    init_session_state()
    st.title("PDF Chatbot (向量增强检索)")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    display_chat_history()

    if uploaded_file is not None:
        file_path = save_pdf(uploaded_file)
        try:
            with st.status("Analyzing your document and updating vector store..."):
                process_pdf(file_path)
        except Exception as e:
            st.error(f"向量化失败: {e}")
            return
        qa_chain = get_qa_chain()
        if user_input := st.chat_input("You:", key="user_input"):
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    message_placeholder = st.empty()
                    stream_handler = StreamlitCallbackHandler(message_placeholder)
                    # 重新实例化 LLM，传入自定义 handler
                    llm = OllamaLLM(
                        base_url=OLLAMA_LLM_BASE_URL,
                        model=OLLAMA_LLM_MODEL,
                        verbose=True,
                        callbacks=[stream_handler],
                    )
                    # 重新实例化 qa_chain，传入新的 llm
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(),
                        verbose=True,
                        chain_type_kwargs={
                            "verbose": True,
                            "prompt": st.session_state.prompt,
                            "memory": st.session_state.memory,
                        },
                    )
                    response = qa_chain(user_input)
                    # 最终输出
                    message_placeholder.markdown(stream_handler.tokens)
            chatbot_message = {"role": "assistant", "message": stream_handler.tokens}
            st.session_state.chat_history.append(chatbot_message)
    else:
        st.write("Please upload a PDF file.")


if __name__ == "__main__":
    main()
