---
layout: post
title: Ontario Curriculum Question and Answer Platform (RAG Engine)
image: "/posts/causal-impact-title-img.png"
tags: [RAG, Langchain, Python]
---

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain%20%28LCEL%29-orange)](https://github.com/langchain-ai/langchain)
[![VectorDB: Chroma](https://img.shields.io/badge/VectorDB-Chroma-red)](https://github.com/chroma-core/chroma)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Retrieval-Augmented Generation (RAG) platform engineered to transform dense, unstructured public sector documentation into a high-performance, natural language query engine. 

### 🎯 The Problem & Business Value
The Ontario Ministry of Education publishes sprawling, multi-hundred-page curriculum guidelines across dozens of individual subjects (Mathematics, Language, Science, etc.). Educators are legally and professionally required to navigate these dense documents to build compliant yearly lesson plans—a process that drains hundreds of hours of administrative focus.

This platform acts as a production-ready proof of concept (focusing on the **Language** and **Mathematics** curricula) that allows educators to instantly extract semantic definitions, grading criteria, and overarching guidelines through natural language queries. The underlying data pipeline is completely agnostic and built to scale horizontally across thousands of documents.

---

## 🏗️ System Architecture & Engineering Choices

[Unstructured PDFs] ➔ [PyPDF Data Pipeline] ➔ [Recursive Chunking]
│
[OpenAI gpt-4o-mini] 🔀 [LCEL Chain] 🔀 [Normalized Relevance Filter] 🔀 [Chroma DB]

### 1. Data Processing & Chunking
* **Pipeline Ingestion:** Implemented an automated `PyPDFDirectoryLoader` pipeline that handles directory-wide ingestion of unstructured PDF documents, successfully compiling thousands of structural elements across **250+ source pages**.
* **Semantic Windowing:** Utilized a `RecursiveCharacterTextSplitter` configured with a strict `chunk_size` of 1,000 characters and a `chunk_overlap` of 500 characters. This optimization prevents contextual fracturing across pagination breaks, maintaining the relationship between foundational guidelines and grade-specific sub-clauses, splitting the documentation into **841 highly dense vector nodes**.

### 2. Vector Database and Relevance Filtering
* **Cost-Efficient Embeddings:** Integrated OpenAI’s `text-embedding-3-small` vector model. By naming this explicitly over legacy defaults, API token overhead was reduced by **80%** while simultaneously improving cross-lingual and semantic matching metrics.
* **Normalized Relevance Thresholds:** Replaced standard similarity searches with `similarity_search_with_relevance_scores(k=3)`. This enforces mathematical constraint checking, returning strictly normalized confidence intervals ($[0,1]$ space) to prevent the LLM from processing irrelevant background noise or out-of-bounds context.

### 3. LLM Integration and Orchestration (LCEL)
* **Deterministic Inference:** Built the generation chain utilizing **LangChain Expression Language (LCEL)**, piping the prompt template, a temperature-zero `gpt-4o-mini` model, and a deterministic `StrOutputParser` into a unified execution block. 
* **Strict Context Enforcement:** The engine's structural system prompt forces the LLM to restrict its answers *strictly* to retrieved source context, preventing hallucinations and ensuring high-fidelity educational alignment.

---

## 🚀 Technical Stack

* **Orchestration:** LangChain, LangChain Expression Language (LCEL)
* **Vector Database:** Chroma DB (Local vector persistence layer)
* **LLM / Embeddings API:** OpenAI (`gpt-4o-mini` / `text-embedding-3-small`)
* **Environment & Tools:** Python, Jupyter, Streamlit Cloud Deployment, Object-Oriented System Design

---

## 🎯 Future Scalability Roadmap
* **Multi-Modal Parsing:** Upgrading document parsing pipelines to process scanned diagrams, matrix tables, and mathematical evaluation blocks via vision-based ingestion tools.
* **Hybrid Search Strategy:** Introducing sparse keyword retrieval (BM25) side-by-side with dense semantic embeddings to optimize specific numeric curriculum code searches (e.g., searching specifically for "B1.2").
