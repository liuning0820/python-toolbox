import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
OLLAMA_MODEL = "qwen3:32b"
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://localhost:11434")

st.title("Web Page Summarizer with Ollama")

url = st.text_input("Paste the URL of the web page you want to summarize:")

def fetch_webpage_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract visible text from the page
        texts = soup.stripped_strings
        return " ".join(texts)
    except Exception as e:
        st.error(f"Error fetching the web page: {e}")
        return None

if st.button("Summarize"):
    if url:
        text = fetch_webpage_text(url)
        if text:
            prompt = PromptTemplate(
                input_variables=["content"],
                template="Summarize the following content:\n\n{content}\n\nSummary:"
            )
            llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_LLM_BASE_URL
            )
            chain = prompt | llm
            with st.spinner("Summarizing..."):
                summary = chain.invoke({"content": text})
            st.subheader("Summary:")
            st.write(summary)
    else:
        st.warning("Please enter a valid URL.")