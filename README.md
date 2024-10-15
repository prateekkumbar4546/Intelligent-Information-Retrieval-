## RAG-based Question Answering System
This repository contains the implementation of a RAG-based Question Answering System using FastAPI for serving the model and generating text and audio responses based on user queries. It integrates a Retrieval-Augmented Generation (RAG) system and an agent system to handle complex queries. Additionally, it includes Text-to-Speech (TTS) functionality to provide audio responses.

Features
1. RAG System: Uses Retrieval-Augmented Generation to answer queries by retrieving relevant documents and generating responses.
2. Agent System: Orchestrates the interaction between the RAG system and other components.
3. Text-to-Speech (TTS): Converts generated text responses into speech using Sarvam TTS.
4. FastAPI: The application is built with FastAPI, providing both HTML and JSON-based responses.
5. Logging: Logs key events such as received queries and responses.
