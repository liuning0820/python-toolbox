# rag_chat.py
import chromadb
from ollama import embed, chat

DB_DIR = "rag_vectors_db"

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("notes")

MODES = {
    "1": "总结模式",
    "2": "提问模式",
    "3": "洞察模式"
}

def build_prompt(mode: str, context: str, query: str) -> str:
    if mode == "1":  # 总结
        return f"""你是一个学习助手。以下是我的笔记片段：
{context}

请帮我用中文总结这些内容，提炼出关键点。"""
    elif mode == "2":  # 提问
        return f"""你是一个本地知识助手。以下是我的笔记片段：
{context}

基于这些内容，回答我的问题：{query}"""
    elif mode == "3":  # 洞察
        return f"""你是一个洞察助手。以下是我的笔记片段：
{context}

请帮我提炼关键见解，指出潜在联系，给出新的思考方向。"""
    else:
        return f"""以下是我的笔记片段：
{context}

基于这些内容，回答问题：{query}"""

def rag_query(query: str, mode: str):
    query_emb = embed(model="bge-m3", input=query).embeddings[0]
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    context = "\n\n".join(results["documents"][0])
    prompt = build_prompt(mode, context, query)

    response = chat(model="qwen2.5-coder:1.5b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    print("请选择模式：")
    for k, v in MODES.items():
        print(f"{k}. {v}")
    mode = input("输入数字选择: ").strip()

    while True:
        q = input("\n❓ 问题: ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        print("\n💡 回答:\n", rag_query(q, mode))
