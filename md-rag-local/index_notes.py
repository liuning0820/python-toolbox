# index_notes.py
import os
import chromadb
from markdown_it import MarkdownIt
from ollama import embed
from dotenv import load_dotenv

load_dotenv()

NOTES_DIR = "notes"
DB_DIR = os.environ.get("DB_DIR", "rag_vectors_db")

md = MarkdownIt()

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("notes")

def split_text(text, max_length=500):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i+max_length])

def index_notes():
    for filename in os.listdir(NOTES_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(NOTES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = list(split_text(content))
            for i, chunk in enumerate(chunks):
                emb = embed(model="bge-m3", input=chunk).embeddings[0]
                collection.add(
                    documents=[chunk],
                    embeddings=[emb],
                    ids=[f"{filename}-{i}"]
                )
            print(f"Indexed {filename}, {len(chunks)} chunks.")

if __name__ == "__main__":
    index_notes()
    print("✅ All notes indexed.")
