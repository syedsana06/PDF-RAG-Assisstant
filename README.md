# 📚 PDF RAG Assistant

## PDF-Based RAG Question Answering System

A Retrieval-Augmented Generation application that allows users to upload PDF documents and ask questions about their contents.

The application retrieves relevant information from the uploaded documents using semantic search and then sends only the retrieved context to a Groq-powered Large Language Model.

---

# Problem Statement

Large PDF documents can contain hundreds of pages, making it difficult for users to quickly locate relevant information.

Sending the complete PDF to an LLM is inefficient and can exceed context limits.

Retrieval-Augmented Generation solves this problem by:

1. Extracting document text.
2. Splitting text into smaller chunks.
3. Converting chunks into embeddings.
4. Storing embeddings in a vector database.
5. Converting the user's question into an embedding.
6. Searching for similar document chunks.
7. Sending only relevant chunks to the LLM.
8. Generating a grounded answer.

---

# Objective

The objective of this project is to implement a complete PDF-based RAG question-answering system.

The system should provide:

- PDF upload
- PDF text extraction
- Page metadata
- Text chunking
- Sentence Transformer embeddings
- FAISS vector search
- Groq LLM generation
- Grounded answers
- Source references
- Page references
- Conversation history
- Multiple PDF support

---

# Features

## PDF Upload

Users can upload one or multiple PDF files.

The application extracts text page-by-page using PyMuPDF.

---

## Text Chunking

LangChain's RecursiveCharacterTextSplitter is used.

Two predefined configurations are available:

### Configuration A

```text
Chunk size: 500
Overlap: 50