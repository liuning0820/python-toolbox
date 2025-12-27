# RAG (Retrieval-Augmented Generation) with Markdown Notes

This project demonstrates how to use RAG techniques to enhance the capabilities of language models by indexing and retrieving information from a set of Markdown notes.

```sh

# 通过环境变量指定笔记目录或默认目录
python index_notes.py

python index_notes.py ./notes

# 指定任意笔记目录（支持绝对路径或相对路径）
python watch_notes.py /path/to/your/notes

# 启动 RAG 聊天CLI
python rag_chat.py

# 简易 Web UI
streamlit run rag_app.py

```
