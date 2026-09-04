# easy-rag · 中医名家知识问答的 RAG 流水线

**一条从零手写的 RAG 流水线。重点不在"能跑"，而在把中间的两个旋钮拧出来单独做：怎么切文档，怎么召回。**

## 解决什么问题

大多数 RAG 教程止步于"能问出答案"。这个项目把分块与检索两步各自拆成可替换的策略，并配一套离线评测脚本，让"哪种配置更好"变成有数字的结论，而不是感觉。语料是几位上海名中医的学术思想与临证经验文本（`sources/`，9 篇），问答界面用 Streamlit。

## 流水线

| 阶段 | 模块 | 做了什么 |
|---|---|---|
| 预处理 | `preprocess.py` → `data/processed_data.json` | 从 txt 抽标题与正文，切成带 `source_file` / `chunk_index` 的段落 |
| 分块优化 | `chunk_optimizer.py` | `smart_chunk_text` 按语义边界切分，`enhance_document_metadata` / `extract_document_keywords` 给每块补关键词元数据 |
| 向量化与索引 | `models.py`、`chroma_utils.py` | `moka-ai/m3e-base` 中文嵌入，ChromaDB 持久化集合，首次运行自动建索引 |
| 检索优化 | `retrieval_optimizer.py` | `hybrid_search`（向量 + 关键词）、`query_expansion`、`rerank_documents`、`remove_duplicate_documents`、`multi_hop_retrieval` |
| 生成 | `rag_core.py` | 拼上下文交给 DeepSeek API 生成（`config.py` 里换模型） |
| 评测 | `evaluate_rag.py`、`test_rag_performance.py` | `evaluate_retrieval_relevance` 与 `evaluate_generation_quality` 两组指标，`run_evaluation` 一键跑 |

## 快速开始

```bash
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 国内下载嵌入模型用（见 run.sh）
# 在 config.py 里填入 DEEPSEEK_API_KEY
python preprocess.py        # 由 sources/ 生成 data/processed_data.json
streamlit run app.py        # 交互问答
python evaluate_rag.py      # 离线评测
python diagnostics.py       # 环境与索引自检
```

## 技术栈

Python · Streamlit · ChromaDB · sentence-transformers（m3e-base）· DeepSeek API

## 已知不足

- 语料很小（9 篇），评测集为自建，结论只在本数据集上成立。
- `config.py` 里还留着早期 Milvus Lite 的参数，当前实际使用 ChromaDB；`requirements.txt` 中的 `pymilvus` 同属遗留。
- 生成侧只接了 DeepSeek，换本地 Ollama 需改 `models.py`。

个人练习项目（2026-02）。
