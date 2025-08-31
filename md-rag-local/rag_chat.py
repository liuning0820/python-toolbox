# rag_chat.py
import chromadb
from ollama import embed, chat

DB_DIR = "rag_vectors_db"

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("notes")

def rag_query(query: str):
    query_emb = embed(model="bge-m3", input=query).embeddings[0]
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    context = "\n\n".join(results["documents"][0])
    prompt = f"""你是一个本地知识助手。以下是我的笔记片段：
{context}

基于这些内容，回答问题：{query}
"""

    response = chat(model="qwen2.5-coder:1.5b", messages=[{"role":"user","content":prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    while True:
        q = input("\n❓ 问题: ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        print("\n💡 回答:\n", rag_query(q))
