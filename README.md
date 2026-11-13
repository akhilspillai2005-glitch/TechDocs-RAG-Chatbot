# TechDocs AI – Python Documentation RAG Chatbot

## Project Overview
TechDocs AI is a Retrieval-Augmented Generation (RAG) chatbot that answers Python programming questions using documentation.

## Technologies Used
- Python
- Streamlit
- LangChain
- Hugging Face Embeddings
- FAISS Vector Database
- Google Gemini API

## RAG Workflow
Document Loading → Text Splitting → Embeddings → FAISS Vector Database → Retriever → Gemini LLM → Final Response

## Features
- Interactive chatbot interface
- Python documentation-based answers
- Semantic search using FAISS
- AI-generated explanations and code examples
- Retrieved source references
- Gemini model fallback support

## How to Run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_api_key
```

4. Start the application:

```bash
streamlit run app.py
```

## Developer
Akhil S Pillai

## GitHub
https://github.com/akhilspillai2005-glitch/TechDocs-RAG-Chatbot
