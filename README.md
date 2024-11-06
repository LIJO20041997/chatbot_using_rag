# Chatbot for Websites using RAG (Retrieval-Augmented Generation)

This Streamlit application creates a chatbot capable of answering questions based on content retrieved from any specified URL. The app uses LangChain for RAG (Retrieval-Augmented Generation), Chroma for document vector storage, and OpenAI's language model for response generation.

## Features

- Accepts a URL as input, loads content from the webpage, and splits it into manageable chunks.
- Creates a vectorized representation of the document chunks to allow efficient retrieval.
- Uses a question-answering model that retrieves relevant document chunks to generate answers.
- Concise responses that are limited to three sentences for brevity.

## Requirements

- Python 3.8+
- Required Python packages listed in [requirements.txt](#requirements)

## Setup and Installation
- git clone https://github.com/LIJO20041997/chatbot_using_rag.git
- cd chatbot_using_rag
- streamlit run app.py
