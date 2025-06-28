import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
OLLAMA_LLM_MODEL=os.getenv("OLLAMA_LLM_MODEL", "qwen3:32b")
OLLAMA_EMBEDDING_MODEL=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_BASE_URL = os.getenv('OLLAMA_EMBEDDING_BASE_URL', "http://localhost:11434")
VECTOR_DIR = 'web_vectors'

# Ensure the vector directory exists
os.makedirs(VECTOR_DIR, exist_ok=True)


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler to stream LLM responses to a Streamlit placeholder."""

    def __init__(self, message_placeholder):
        self.message_placeholder = message_placeholder
        self.tokens = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += token
        self.message_placeholder.markdown(self.tokens + "▌")

    def on_llm_end(self, response, **kwargs):
        self.message_placeholder.markdown(self.tokens)


def init_session_state():
    """Initializes the session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_url' not in st.session_state:
        st.session_state.processed_url = None
    if 'template' not in st.session_state:
        st.session_state.template = (
            "You are a knowledgeable chatbot. Use the following context from a web page to answer the user's question. "
            "Your tone should be professional and informative.\n\n"
            "Context: {context}\nHistory: {history}\n\nUser: {question}\nChatbot:"
        )
    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template,
        )
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history", return_messages=True, input_key="question"
        )


def fetch_webpage_text(url):
    """Fetches and extracts text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return " ".join(soup.stripped_strings)
    except Exception as e:
        st.error(f"Error fetching the web page: {e}")
        return None


def setup_chat_environment(url, text_content):
    """Processes text, creates vector store, and generates an initial summary."""
    with st.status("Analyzing web page and preparing for chat..."):
        # 1. Split text into chunks
        st.write("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200, length_function=len
        )
        splits = text_splitter.split_text(text_content)

        # 2. Create vector store
        st.write("Creating vector store...")
        embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_EMBEDDING_BASE_URL)
        vectorstore = Chroma.from_texts(
            texts=splits,
            embedding=embedding_function,
            persist_directory=f"{VECTOR_DIR}/{os.path.basename(url)}"
        )
        st.session_state.vectorstore = vectorstore

        # 3. Generate initial summary
        st.write("Generating initial summary...")
        summary_prompt = PromptTemplate.from_template("Summarize the following content:\n\n{content}")
        llm = OllamaLLM(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_LLM_BASE_URL)
        chain = summary_prompt | llm
        summary = chain.invoke({"content": text_content[:4000]})  # Summarize first 4000 chars

        st.session_state.chat_history.append({"role": "assistant",
                                              "message": f"I have summarized the content from the URL. Here is a brief overview:\n\n{summary}\n\nYou can now ask me questions about it."})
        st.session_state.processed_url = url


st.title("Web Page Summarizer and Chatbot")

init_session_state()

url = st.text_input("Paste the URL of the web page you want to analyze:", key="url_input")

if url and url != st.session_state.processed_url:
    if st.button("Analyze and Start Chat"):
        # Reset state for new URL
        st.session_state.chat_history = []
        st.session_state.memory.clear()

        text = fetch_webpage_text(url)
        if text:
            setup_chat_environment(url, text)
            st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Handle chat input if a URL has been processed
if st.session_state.processed_url:
    if user_input := st.chat_input("Ask a question about the web page:", key="user_chat_input"):
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            stream_handler = StreamlitCallbackHandler(message_placeholder)

            # Create LLM with streaming callback
            llm = OllamaLLM(
                model=OLLAMA_LLM_MODEL,
                base_url=OLLAMA_LLM_BASE_URL,
                callbacks=[stream_handler]
            )

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=st.session_state.vectorstore.as_retriever(),
                chain_type_kwargs={
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )

            qa_chain.invoke(user_input)
            response_message = stream_handler.tokens

        st.session_state.chat_history.append({"role": "assistant", "message": response_message})