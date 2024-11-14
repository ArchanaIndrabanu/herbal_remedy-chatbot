import streamlit as st
from pymongo import MongoClient, errors
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
from datetime import datetime

# MongoDB setup
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    db = client['herbal_remedy_chatbot']
    context_collection = db['context']
    sessions_collection = db['sessions']
    st.sidebar.success("Connected to MongoDB successfully")
except errors.ServerSelectionTimeoutError as err:
    st.sidebar.error("Failed to connect to MongoDB: {}".format(err))

# Load documents from directory
def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents and chunk them
directory_path = "./herbal"
documents = load_documents_from_directory(directory_path)

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# VectorStore setup using MongoDB
class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    def populate_vectors(self, dataset):
        for doc in dataset:
            if context_collection.find_one({"id": doc["id"]}) is None:
                doc["embedding"] = self.embedding_model.encode(doc["text"]).tolist()
                try:
                    context_collection.insert_one({
                        "id": doc["id"],
                        "text": doc["text"],
                        "embedding": doc["embedding"]
                    })
                except Exception as e:
                    st.sidebar.error(f"Error inserting document {doc['id']} into MongoDB: {e}")

    def search_context(self, query, n_results=3):
        query_embedding = self.embedding_model.encode(query).tolist()
        contexts = list(context_collection.find().limit(100))
        contexts.sort(key=lambda x: self.cosine_similarity(query_embedding, x['embedding']), reverse=True)
        return contexts[:n_results]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return sum(a * b for a, b in zip(vec1, vec2)) / (sum(a ** 2 for a in vec1) ** 0.5 * sum(b ** 2 for b in vec2) ** 0.5)

# Instantiate VectorStore and populate vectors
vector_store = VectorStore()
vector_store.populate_vectors(chunked_documents)

# Using a model that generates concise responses
class SmallInstructModel:
    def __init__(self):
        self.model = pipeline("text-generation", model="distilgpt2")

    def generate_answer(self, question, context=None):
        if context is None:
            context = ""

        prompt = f"Question: {question}\nContext: {context}\nAnswer: "

        max_length = 100
        truncated_prompt = " ".join(prompt.split()[:max_length])

        response = self.model(truncated_prompt, max_new_tokens=50, num_return_sequences=1, do_sample=True)[0]["generated_text"].strip()

        # Check if the response is complete
        if not response.endswith("."):
            # Recursively generate a new response if the previous one was incomplete
            return self.generate_answer(question, context)
        else:
            return response

# Initialize the small model
small_model = SmallInstructModel()

def generate_conversation_id():
    """Generate a unique conversation ID using timestamp"""
    return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Database operations
def save_or_update_user_session(user_id, question, answer, context, conversation_id):
    """Save or update the conversation in MongoDB"""
    session = sessions_collection.find_one({
        "user_id": user_id,
        "conversation_id": conversation_id
    })
    
    message = {
        "question": question,
        "answer": answer,
        "context": context,
        "timestamp": datetime.now()
    }
    
    if session:
        sessions_collection.update_one(
            {"user_id": user_id, "conversation_id": conversation_id},
            {"$push": {"conversation_history": message}}
        )
    else:
        sessions_collection.insert_one({
            "user_id": user_id,
            "conversation_id": conversation_id,
            "created_at": datetime.now(),
            "conversation_history": [message]
        })

def retrieve_user_sessions(user_id, conversation_id):
    """Retrieve conversation history from MongoDB"""
    session = sessions_collection.find_one({
        "user_id": user_id,
        "conversation_id": conversation_id
    })
    return session["conversation_history"] if session else []

def get_all_user_conversations(user_id):
    """Get all conversation IDs for a user with timestamps"""
    conversations = list(sessions_collection.find(
        {"user_id": user_id},
        {"conversation_id": 1, "created_at": 1}
    ).sort("created_at", -1))
    return conversations

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'conversation_id' not in st.session_state:
        st.session_state['conversation_id'] = None
    if 'chat_active' not in st.session_state:
        st.session_state['chat_active'] = False
    if 'current_conversation' not in st.session_state:
        st.session_state['current_conversation'] = []
    if 'needs_rerun' not in st.session_state:
        st.session_state['needs_rerun'] = False

def handle_send():
    """Handle sending a message"""
    if st.session_state.user_input:
        # Get response
        contexts = vector_store.search_context(st.session_state.user_input)
        context_text = " ".join(ctx['text'] for ctx in contexts)
        response = small_model.generate_answer(st.session_state.user_input, context_text)
        
        # Save to database
        save_or_update_user_session(
            st.session_state.user_id,
            st.session_state.user_input,
            response,
            context_text,
            st.session_state['conversation_id']
        )
        
        # Update session state
        st.session_state['current_conversation'].append({
            "question": st.session_state.user_input,
            "answer": response,
            "context": context_text,
            "timestamp": datetime.now()
        })
        
        # Clear input
        st.session_state.user_input = ""
        st.session_state.needs_rerun = True

def handle_start_new_conversation():
    """Handler for starting a new conversation"""
    new_conv_id = generate_conversation_id()
    st.session_state['conversation_id'] = new_conv_id
    st.session_state['current_conversation'] = []
    st.session_state['chat_active'] = True
    st.session_state.needs_rerun = True

def handle_select_conversation(conv_id):
    """Handler for selecting an existing conversation"""
    if st.session_state['conversation_id'] != conv_id:
        st.session_state['conversation_id'] = conv_id
        st.session_state['current_conversation'] = retrieve_user_sessions(
            st.session_state.user_id, 
            conv_id
        )
        st.session_state['chat_active'] = True
        st.session_state.needs_rerun = True

# Initialize session state
initialize_session_state()

# Streamlit UI
st.title("Herbal Remedy Chatbot")

# Sidebar
with st.sidebar:
    # User ID input
    user_id = st.text_input("User ID", value="user123")
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.session_state.conversation_id = None
        st.session_state.current_conversation = []
        st.session_state.needs_rerun = True
    
    # New conversation button
    if st.button("Start New Conversation"):
        handle_start_new_conversation()
    
    # Existing conversations
    st.write("Previous Conversations:")
    conversations = get_all_user_conversations(user_id)
    
    for conv in conversations:
        conv_id = conv['conversation_id']
        created_at = conv.get('created_at', datetime.now()).strftime('%Y-%m-%d %H:%M')
        if st.button(f"Conversation from {created_at}", key=conv_id):
            handle_select_conversation(conv_id)

# Main chat interface
if st.session_state['chat_active'] and st.session_state['conversation_id']:
    st.write(f"Current Conversation ID: {st.session_state['conversation_id']}")
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        if st.session_state['current_conversation']:
            for msg in st.session_state['current_conversation']:
                st.write(f"User: {msg['question']}")
                st.write(f"Bot: {msg['answer']}")
                st.write("---")
        
        # Chat input and controls
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.text_input(
                "Ask your question:",
                key="user_input",
                on_change=handle_send
            )
        
        with col2:
            if st.button("End Chat"):
                st.session_state['chat_active'] = False
                st.session_state['conversation_id'] = None
                st.session_state['current_conversation'] = []
                st.session_state.needs_rerun = True

else:
    st.write("Select 'Start New Conversation' or choose an existing conversation from the sidebar.")

# Handle rerun at the end of the script
if st.session_state.needs_rerun:
    st.session_state.needs_rerun = False
    st.rerun()
