import os
import time
import logging

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """
    Ensure that required directories exist.
    Creates FILES_DIR and VECTOR_DIR if they don't exist.
    """
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)


def init_session_state():
    """
    Initialize Streamlit session state variables.
    Sets up template, prompt, memory, vectorstore, LLM, and chat history if they don't exist.
    """
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
    """
    Save uploaded PDF file to the files directory.

    Args:
        uploaded_file: The uploaded file object from Streamlit

    Returns:
        str: The path to the saved file

    Raises:
        ValueError: If the file is not a PDF
        IOError: If there's an error saving the file
    """
    # Validate file type
    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")

    filename = uploaded_file.name
    file_path = os.path.join(FILES_DIR, filename)

    # Check if file already exists with the same content using hash comparison
    if os.path.exists(file_path):
        try:
            # Calculate hash of uploaded file
            uploaded_hash = calculate_file_hash(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer

            # Calculate hash of existing file
            with open(file_path, "rb") as existing_file:
                existing_hash = calculate_file_hash(existing_file)

            # If hashes are the same, return existing file path
            if uploaded_hash == existing_hash:
                logger.info(f"File already exists with same content: {file_path}")
                return file_path

            # If content is different, handle filename conflict
            counter = 1
            original_filename = filename
            while os.path.exists(file_path):
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                file_path = os.path.join(FILES_DIR, filename)
                counter += 1

        except Exception as e:
            logger.warning(f"Could not compare file content, treating as new file: {e}")
            # Handle filename conflict
            counter = 1
            original_filename = filename
            while os.path.exists(file_path):
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                file_path = os.path.join(FILES_DIR, filename)
                counter += 1

    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        logger.info(f"Saved PDF file: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving PDF file: {e}")
        raise IOError(f"Failed to save PDF file: {e}")


def process_pdf(file_path):
    """
    Process a PDF file by loading, splitting, and vectorizing its content.
    Only adds documents to the vector store if they haven't been processed before.

    Args:
        file_path (str): Path to the PDF file to process

    Raises:
        Exception: If there's an error processing the PDF
    """
    try:
        logger.info(f"Processing PDF: {file_path}")

        # Load PDF content
        loader = PyPDFLoader(file_path)
        data = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200, length_function=len
        )
        all_splits = text_splitter.split_documents(data)

        # Load existing vector store
        vectorstore = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=OllamaEmbeddings(
                base_url=OLLAMA_EMBEDDING_BASE_URL, model=OLLAMA_EMBEDDING_MODEL
            ),
        )

        # Check if this file has already been processed
        metadatas = vectorstore.get().get("metadatas", [])
        sources = [meta.get("source", "") for meta in metadatas if isinstance(meta, dict)]

        if file_path not in sources:
            # Deduplicate documents before adding
            unique_splits = deduplicate_documents(all_splits, sources)
            logger.info(f"Adding {len(unique_splits)} unique chunks to vector store")
            vectorstore.add_documents(unique_splits)
        else:
            logger.info(f"PDF already processed, skipping vectorization")

        st.session_state.vectorstore = vectorstore

    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        raise Exception(f"Failed to process PDF: {e}")


def get_qa_chain():
    """
    Get or create the QA chain for answering questions.

    Returns:
        RetrievalQA: The QA chain for answering questions based on the vector store
    """
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
    """
    Display the chat history in the Streamlit UI.
    """
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
    """
    Main function to run the PDF chatbot application.
    Sets up the UI, handles file uploads, and manages the chat interface.
    """
    try:
        ensure_dirs()
        init_session_state()
        st.title("PDF Chatbot (向量增强检索)")
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
        display_chat_history()

        if uploaded_file is not None:
            try:
                file_path = save_pdf(uploaded_file)
                with st.status("Analyzing your document and updating vector store..."):
                    process_pdf(file_path)
            except ValueError as e:
                st.error(f"File upload error: {e}")
                return
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
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

                        # Use existing LLM and QA chain from session state
                        # Create a new LLM with the stream handler for streaming output
                        llm_with_stream = OllamaLLM(
                            base_url=OLLAMA_LLM_BASE_URL,
                            model=OLLAMA_LLM_MODEL,
                            verbose=True,
                            callbacks=[stream_handler],
                        )

                        # Create a new QA chain with the streaming LLM
                        qa_chain_with_stream = RetrievalQA.from_chain_type(
                            llm=llm_with_stream,
                            chain_type="stuff",
                            retriever=st.session_state.vectorstore.as_retriever(),
                            verbose=True,
                            chain_type_kwargs={
                                "verbose": True,
                                "prompt": st.session_state.prompt,
                                "memory": st.session_state.memory,
                            },
                        )

                        try:
                            response = qa_chain_with_stream(user_input)
                            # 最终输出
                            message_placeholder.markdown(stream_handler.tokens)
                        except Exception as e:
                            logger.error(f"Error generating response: {e}")
                            st.error("Sorry, I encountered an error while generating a response.")
                            message_placeholder.markdown("I'm sorry, but I couldn't generate a response.")

                chatbot_message = {"role": "assistant", "message": stream_handler.tokens}
                st.session_state.chat_history.append(chatbot_message)
        else:
            st.write("Please upload a PDF file.")
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        st.error(f"An unexpected error occurred: {e}")

def calculate_file_hash(file_obj, chunk_size=8192):
    """
    Calculate SHA-256 hash of a file object.

    Args:
        file_obj: File object to calculate hash for
        chunk_size: Size of chunks to read (default: 8192 bytes)

    Returns:
        str: SHA-256 hash of the file content
    """
    import hashlib

    sha256 = hashlib.sha256()

    # Read file in chunks to handle large files efficiently
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            break
        sha256.update(chunk)

    return sha256.hexdigest()

def deduplicate_documents(documents, existing_sources):
    """
    Remove duplicate documents based on content hash.

    Args:
        documents: List of document objects to process
        existing_sources: List of existing source files in vector store

    Returns:
        List of unique documents to add to vector store
    """
    unique_docs = []
    seen_hashes = set()

    for doc in documents:
        # Create a hash of the document content
        import hashlib
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()

        # Only add if we haven't seen this content before
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    return unique_docs

if __name__ == "__main__":
    main()
