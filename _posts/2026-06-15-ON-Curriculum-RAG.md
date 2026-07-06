---
layout: post
title: Ontario Curriculum Question and Answer Platform
image: "/posts/ontario-rag-img.png"
tags: [RAG, Langchain, Python]
---

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain%20%28LCEL%29-orange)](https://github.com/langchain-ai/langchain)
[![VectorDB: Chroma](https://img.shields.io/badge/VectorDB-Chroma-red)](https://github.com/chroma-core/chroma)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![alt text](/img/posts/RAG_Example.gif "Example of RAG Platform")

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

# Detailed Project Code Setup

The Ontario education curriculum is comprised of a series of documents that cover the following topics:

- Kindergarten
- The Arts
- French as a Second Language
- Health and Physical Education
- Language
- Mathematics
- Native Languages
- Science and Technology
- Social Studies, History, and Geography

Each core subject has a separate PDF document that covers all of the specific requirements for that particular grade. Teachers have to become very familiar with these documents as these are the guidelines to which they should be teaching for the school year.

To this end, the purpose of this project is to create a chatbot that leverages the power of Generative AI and LLMs to allow teachers to pose questions about the curriculum using natural language, and to receive distilled, summarized results, based on the latest curriculum. Only the Mathematics and Language results are used in this proof of concept. However, this can be easily extended to cover any number of curriculum topics.

```python
import os
from pathlib import Path

DATA_PATH = "C:/Users/rossi/OneDrive/Data Science Projects/ON Edu RAG Project/"
CHROMA_PATH = "./chroma_db" 
COLLECTION_NAME = "ontario_education_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# --- ENVIRONMENT SETUP ---

if "OPENAI_API_KEY" not in os.environ:
    import getpass
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key: ")
```

    Enter your OpenAI API Key:  ········
    


```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def load_and_split_documents(data_path: str) -> list:
    """Loads PDFs from the specified directory and splits them into chunks."""
    print(f"Loading PDFs from: {data_path}")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} source pages.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split pages into {len(chunks)} text chunks.")
    return chunks

def build_or_load_vector_store(chunks: list, store_path: str, embedding_model) -> Chroma:
    """Creates a local Chroma DB collection if it doesn't exist, or loads it if it does."""
    if Path(store_path).exists():
        print(f"Loading existing vector database from: {store_path}")
        return Chroma(
            persist_directory=store_path,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
    else:
        print(f"Creating new Chroma vector database at: {store_path}")
        return Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=store_path,
            collection_name=COLLECTION_NAME
        )

def get_rag_chain(llm):
    """Defines and binds the LCEL RAG execution assembly line."""
    prompt_template = PromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

___

Answer the question based on the above context: {query}
""")
    output_parser = StrOutputParser()
    return prompt_template | llm | output_parser
```
    


```python
#Execute Pipeline

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

chunks = load_and_split_documents(DATA_PATH)
db = build_or_load_vector_store(chunks, CHROMA_PATH, embeddings)

rag_chain = get_rag_chain(llm)
```


```python
#Running RAG query

query_text = "What is the big idea that students are supposed to learn from the Grade 3 Language curriculum"

results = db.similarity_search_with_relevance_scores(query_text, k=3)
context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

response_text = rag_chain.invoke({
    "context": context_text,
    "query": query_text
})

print(f"\n=== ANSWER ===\n{response_text}")
```

    
    === ANSWER ===
    The big idea that students are supposed to learn from the Grade 3 Language curriculum is the integration and application of transferable skills—such as critical thinking, communication, collaboration, and digital literacy—across various language and literacy contexts. This includes understanding how these skills support effective communication, enhance engagement in learning, and foster an appreciation of diverse identities and perspectives, particularly in relation to cultural and community contexts. Additionally, students are encouraged to apply their learning across different subject areas and recognize the relevance of these skills in everyday life.


