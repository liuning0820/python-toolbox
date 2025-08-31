# watch_notes.py
import time
import os
import chromadb
from markdown_it import MarkdownIt
from ollama import embed
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

NOTES_DIR = "notes"
DB_DIR = "rag_vectors_db"

md = MarkdownIt()
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("notes")

def split_text(text, max_length=500):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i+max_length])

def index_file(filepath):
    filename = os.path.basename(filepath)
    if not filename.endswith(".md"):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = list(split_text(content))

    # 先删除旧的
    ids_to_delete = [f"{filename}-{i}" for i in range(1000)]
    collection.delete(ids=ids_to_delete)

    # 再添加新的
    for i, chunk in enumerate(chunks):
        emb = embed(model="bge-m3", input=chunk).embeddings[0]
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[f"{filename}-{i}"]
        )
    print(f"✅ {filename} 已重新索引，共 {len(chunks)} 段。")

class NotesHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            index_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            index_file(event.src_path)

if __name__ == "__main__":
    print("👀 正在监听 notes/ 目录的变动...")
    observer = Observer()
    observer.schedule(NotesHandler(), NOTES_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
