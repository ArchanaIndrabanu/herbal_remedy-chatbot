🚀 Project: Herbal Remedy Chatbot 🌿

🔍 Overview: This project showcases a sophisticated AI-powered chatbot designed to provide herbal remedy advice. It leverages MongoDB for efficient data storage and advanced natural language processing models for accurate responses.

💾 Tech Stack:

Streamlit: Interactive web app framework 🖥️
MongoDB: Database for storing context and user sessions 📚
Sentence-Transformers: Embedding model for text similarity 🔍
GPT-J-6B: Language model for generating detailed responses 💬
🔧 Features:

MongoDB Integration: Connects seamlessly to MongoDB for storing and retrieving user sessions and context.
Contextual Responses: Utilizes SentenceTransformer to fetch relevant context from a large dataset.
Advanced Text Generation: Employs GPT-J-6B to generate accurate and context-aware answers.
Session Management: Tracks user interactions and stores conversation history for personalized experience.
Streamlit Interface: Provides a user-friendly interface for interaction and conversation flow.
🌟 How It Works:

Load Documents: Extracts and chunks text files from a specified directory for context.
Populate VectorStore: Embeds text chunks and stores them in MongoDB.
Chatbot Interaction: Retrieves relevant context, generates answers using GPT-J-6B, and processes responses for clarity.
Session Storage: Saves user questions, answers, and context for continuity in conversations.
