# rag_app.py
import streamlit as st
import chromadb
from ollama import embed, chat
from pathlib import Path

from dotenv import load_dotenv
import os


load_dotenv()
DB_DIR = os.environ.get("DB_DIR", "rag_vectors_db")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma3:1b")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3")


client = chromadb.PersistentClient(path=DB_DIR)

# 如果没有 collection，就创建
try:
    collection = client.get_collection("notes")
except:
    collection = client.create_collection("notes")

MODES = {
    "总结模式（带思考问题）": "1",
    "提问模式": "2",
    "洞察模式（带行动步骤）": "3"
}

# ----------------- 工具函数 -----------------
def build_prompt(mode: str, context: str, query: str) -> str:
    if mode == "1":
        return f"""你是一个学习助手。以下是我的笔记片段：
{context}

请帮我用中文总结这些内容，提炼出关键点。"""
    elif mode == "2":
        return f"""你是一个本地知识助手。以下是我的笔记片段：
{context}

基于这些内容，回答我的问题：{query}"""
    elif mode == "3":
        return f"""你是一个洞察助手。以下是我的笔记片段：
{context}

请帮我提炼关键见解，指出潜在联系，给出新的思考方向。"""
    else:
        return f"以下是我的笔记片段：\n{context}\n\n基于这些内容，回答问题：{query}"

def rag_query(query: str, mode: str):
    query_emb = embed(model=EMBEDDING_MODEL, input=query).embeddings[0]
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    context = "\n\n".join(results["documents"][0]) if results["documents"] else "（没有找到相关内容）"
    prompt = build_prompt(mode, context, query)
    response = chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"], context

def generate_thinking_questions(summary: str):
    prompt = f"""基于以下总结内容，生成 3 个中文思考问题，引导深入理解和应用：
{summary}"""
    response = chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def generate_action_items(insights: str):
    prompt = f"""基于以下洞察内容，提出 3-5 个可执行的行动步骤（action items），帮助落实和应用这些洞察：
{insights}"""
    response = chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def add_markdown_to_collection(file, filename: str):
    """读取 markdown 文件，按段落切分，存入 Chroma"""
    text = file.read().decode("utf-8", errors="replace")
    chunks = text.split("\n\n")
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            emb = embed(model=EMBEDDING_MODEL, input=chunk).embeddings[0]
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                ids=[f"{filename}_{i}"]
            )


def clear_database():
    """清空当前数据库"""
    global collection
    client.delete_collection("notes")
    collection = client.create_collection("notes")

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="RAG Chat", page_icon="📚")

st.title("📚 本地 RAG 学习助手")

# ---- 文件上传区 ----
st.sidebar.header("📂 上传 Markdown 笔记")
uploaded_files = st.sidebar.file_uploader("选择 .md 文件", type=["md"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        add_markdown_to_collection(f, f.name)
    st.sidebar.success(f"✅ 已添加 {len(uploaded_files)} 个笔记到数据库")

# 清空数据库按钮
if st.sidebar.button("🗑️ 清空数据库"):
    clear_database()
    st.sidebar.warning("⚠️ 数据库已清空！")

# ---- 模式选择 ----
mode_label = st.radio("选择模式", list(MODES.keys()))
mode = MODES[mode_label]

query = st.text_area("输入你的问题（总结模式可留空）", "")

if st.button("运行 RAG"):
    if mode == "1":  # 总结模式
        # 直接取所有文档
        all_docs = collection.get()["documents"]
        context = "\n\n".join(all_docs) if all_docs else "（没有找到笔记内容）"

        prompt = f"""你是一个学习助手。以下是我的笔记片段：
    {context}

    请帮我用中文总结这些内容，提炼出关键点。"""
        response = chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        st.session_state["summary_answer"] = response["message"]["content"]  # ✅ 存入 session_state

        st.subheader("📝 总结")
        st.write(st.session_state["summary_answer"])

        st.subheader("📑 参考内容")
        st.write(context)

    # 单独的按钮，用之前存的结果
    if mode == "1" and st.button("🧠 生成思考问题"):
        if "summary_answer" in st.session_state:
            st.subheader("🧠 思考问题")
            st.write(generate_thinking_questions(st.session_state["summary_answer"]))
        else:
            st.warning("请先点击运行 RAG 生成总结！")


    else:  # 提问模式 / 洞察模式
        if query.strip():
            answer, context = rag_query(query, mode)
            st.subheader("💡 回答")
            st.write(answer)
            st.subheader("📑 参考内容")
            st.write(context)
            if mode == "3":
                if st.button("🚀 生成行动步骤"):
                    st.subheader("🚀 行动步骤")
                    st.write(generate_action_items(answer))
        else:
            st.warning("请输入问题！")
