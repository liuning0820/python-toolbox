import os
import time

import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings

FILES_DIR = 'files'
VECTOR_DIR = 'jj'

def ensure_dirs():
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

def init_session_state():
    if 'template' not in st.session_state:
        st.session_state.template = (
            "You are a knowledgeable chatbot, here to help with questions of the user. "
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
            memory_key="history",
            return_messages=True,
            input_key="question"
        )
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=OllamaEmbeddings(
                base_url='http://localhost:11434',
                model="bge-m3:latest"
            )
        )
    if 'llm' not in st.session_state:
        st.session_state.llm = OllamaLLM(
            base_url="http://localhost:11434",
            model="gemma3:1b",
            verbose=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def save_pdf(uploaded_file):
    file_path = os.path.join(FILES_DIR, uploaded_file.name + ".pdf")
    if not os.path.isfile(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    return file_path

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    all_splits = text_splitter.split_documents(data)
    st.session_state.vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="bge-m3:latest")
    )

def get_qa_chain():
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.vectorstore.as_retriever(),
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )
    return st.session_state.qa_chain

def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

def main():
    ensure_dirs()
    init_session_state()
    st.title("PDF Chatbot")
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
    display_chat_history()

    if uploaded_file is not None:
        file_path = save_pdf(uploaded_file)
        if not os.path.isfile(os.path.join(VECTOR_DIR, "index")):  # 简单判断是否已向量化
            with st.status("Analyzing your document..."):
                process_pdf(file_path)
        qa_chain = get_qa_chain()
        if user_input := st.chat_input("You:", key="user_input"):
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    response = qa_chain(user_input)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response['result'].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            chatbot_message = {"role": "assistant", "message": response['result']}
            st.session_state.chat_history.append(chatbot_message)
    else:
        st.write("Please upload a PDF file.")

if __name__ == "__main__":
    main()