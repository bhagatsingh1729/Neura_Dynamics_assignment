# Neura_Dynamics_assignment
# Policy Question Answering System (RAG Mini Project)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based Question Answering system over company policy documents (e.g., Shipping, Refund, Cancellation policies).

The system retrieves relevant document chunks using semantic search and generates **grounded, non-hallucinated answers** using a Large Language Model (LLM).

This project was built as part of an **AI Engineer Intern – Take-Home Assignment**, with a focus on:

* Prompt engineering
* Retrieval quality
* Hallucination control
* Clear reasoning and evaluation

---

## Architecture Overview

```
User Question
     ↓
Semantic Retrieval (Chroma + Embeddings)
     ↓
Relevant Policy Chunks
     ↓
Prompt with Strict Instructions
     ↓
LLM (Ollama – llama3.2:1b)
     ↓
Grounded Answer
```

---

## Tech Stack

| Component    | Technology                             |
| ------------ | -------------------------------------- |
| Language     | Python                                 |
| Embeddings   | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB                               |
| LLM          | Ollama (llama3.2:1b)                   |
| Prompting    | Custom RAG Prompt                      |
| Chunking     | RecursiveCharacterTextSplitter         |

---

## Project Structure

```
Internshala_Neura_Dynamics/
│
├── data/
│   └── policy.txt              # Policy documents
│
├── chroma_db/                  # Persisted vector database
│
├── src/
│   ├── load_and_chunk.py       # Data loading & chunking
│   ├── rag_pipeline.py         # RAG logic (retrieve + generate)
│   ├── prompts.py              # Prompt templates
│   └── evaluate.py             # Evaluation questions
│
├── main.py                     # Entry point
├── README.md
└── req.txt                     # Dependencies
```

---

## Data Preparation

* Policy documents are stored as `.txt` files.
* Text is split into chunks of:

  * **Chunk size:** 500 characters
  * **Overlap:** 50 characters

### Why this chunk size?

* Large enough to preserve semantic meaning
* Small enough to avoid irrelevant context
* Optimized for MiniLM embedding model

---

## RAG Pipeline

1. Documents are embedded using **MiniLM embeddings**
2. Embeddings are stored in **ChromaDB**
3. For each query:

   * Top-k relevant chunks are retrieved
   * Retrieved context is injected into the prompt
   * LLM generates answer strictly from context

---

## Prompt Engineering

### Initial Prompt

* Basic instruction to answer from context

### Improved Prompt (Final)

* Explicitly restricts answers to retrieved context
* Handles missing information safely
* Improves clarity and hallucination control

```text
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't know".
```

---

## Evaluation

A small evaluation set is used with:

* Fully answerable questions
* Partially answerable questions
* Unanswerable questions

### Evaluation Criteria

| Metric                  | Method |
| ----------------------- | ------ |
| Accuracy                | Manual |
| Hallucination Avoidance | Manual |
| Clarity                 | Manual |

Scoring:

* ✅ Correct & grounded
* ⚠️ Partial / unclear
* ❌ Hallucinated / incorrect

---

## Edge Case Handling

| Scenario              | System Behavior                 |
| --------------------- | ------------------------------- |
| No relevant documents | Responds with “I don’t know”    |
| Question outside KB   | No hallucination                |
| Partial context       | Partial answer or safe fallback |

---

## LLM Choice (Important Note)

Originally, Hugging Face Inference API was used.
Due to **API endpoint deprecation**, the system was migrated to **Ollama (local LLM)**.

### Why Ollama?

* No external API dependency
* Stable and reproducible
* Embeddings and retrieval remain unchanged
* Demonstrates real-world adaptability

> Changing the LLM does **not affect embeddings or retrieval quality**.

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r req.txt
```

### 3. Start Ollama

```bash
ollama serve
```

Ensure model exists:

```bash
ollama list
```

### 4. Run the Project

```bash
python main.py
```

---

## Key Trade-offs & Future Improvements

### Trade-offs

* Used a lightweight LLM for speed and stability
* Manual evaluation instead of automated metrics

### Improvements With More Time

* Add reranking (Cross-Encoder)
* Add JSON output schema validation
* Add logging & tracing
* Add user interactive Q&A mode
* Use larger instruction-tuned model

---

## Conclusion

This project demonstrates:

* Clear RAG pipeline design
* Strong prompt engineering
* Practical handling of real-world LLM constraints
* Thoughtful evaluation and reasoning




