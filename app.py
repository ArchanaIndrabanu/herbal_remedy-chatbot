import streamlit as st
from pymongo import MongoClient, errors
from sentence_transformers import SentenceTransformer
from transformers import GPTJForCausalLM, GPT2Tokenizer
import os
import re

# MongoDB setup
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)  # 5-second timeout
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
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents and chunk them
directory_path = "./herbal"  # Update this path with your directory
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
            # Check if document is already in MongoDB
            if context_collection.find_one({"id": doc["id"]}) is None:
                # If not, encode and insert the document
                doc["embedding"] = self.embedding_model.encode(doc["text"]).tolist()
                try:
                    context_collection.insert_one({
                        "id": doc["id"],
                        "text": doc["text"],
                        "embedding": doc["embedding"]
                    })
                    st.sidebar.success(f"Document {doc['id']} added to MongoDB")
                except Exception as e:
                    st.sidebar.error(f"Error inserting document {doc['id']} into MongoDB: {e}")

    def search_context(self, query, n_results=3):  # Retrieve top 3 results
        query_embedding = self.embedding_model.encode(query).tolist()
        # Limit number of contexts retrieved to optimize performance
        contexts = list(context_collection.find().limit(100))  # Limit to 100 documents
        contexts.sort(key=lambda x: self.cosine_similarity(query_embedding, x['embedding']), reverse=True)
        return contexts[:n_results]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return sum(a * b for a, b in zip(vec1, vec2)) / (sum(a ** 2 for a in vec1) ** 0.5 * sum(b ** 2 for b in vec2) ** 0.5)

# Instantiate VectorStore and populate vectors only if necessary
vector_store = VectorStore()
vector_store.populate_vectors(chunked_documents)  # Run this only once initially

class AdvancedTextGenerationModel:
    def __init__(self):
        self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def generate_answer(self, question, context=None):
        if context is None:
            context = ""
        # Concatenate the context and question into the input
        input_text = f"Question: {question}\nContext: {context}"
        
        # Encode the input and generate the response
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output_ids = self.model.generate(input_ids, max_new_tokens=150, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=1)[0]
        
        # Decode the output and extract the answer
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return answer

# Process the answer to improve clarity, accuracy, and conciseness
def process_answer(answer):
    """
    Process the model's answer to ensure it's relevant, concise, and accurate.
    """
    # Trim the answer to a single, concise sentence
    answer = answer.split('.')[0] + '.'
    
    # Check if the answer indicates the model doesn't know
    if "I don't know" in answer.lower():
        return "Sorry, I don't have enough information to answer that question."
    
    # Rephrase the answer to be more direct
    answer = re.sub(r"^The answer is", "", answer, flags=re.IGNORECASE).strip()
    
    # Filter out irrelevant information
    answer = re.sub(r"\([^)]*\)", "", answer).strip()
    
    # Ensure the answer is within 1-2 sentences
    if len(answer.split('.')) > 2:
        answer = '. '.join(answer.split('.')[:2]) + '.'
    
    return answer.strip()

# Save user session to MongoDB
def save_user_session(user_id, question, answer, context):
    session = {
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "context": context
    }
    sessions_collection.insert_one(session)

# Retrieve past conversations for a user from MongoDB
def retrieve_user_sessions(user_id):
    sessions = sessions_collection.find({"user_id": user_id})
    past_conversations = ""
    for session in sessions:
        past_conversations += f"Question: {session['question']}\nAnswer: {session['answer']}\nContext: {session['context']}\n\n"
    return past_conversations

# Chatbot interaction function
def chatbot_interaction(user_id, user_question):
    # Retrieve context from MongoDB
    context_response = vector_store.search_context(user_question)
    context = "".join(context['text'] for context in context_response)
    
    # Retrieve past conversations from MongoDB
    past_conversations = retrieve_user_sessions(user_id)
    
    # Generate an answer using the GPT-J-6B model
    generated_answer = advanced_model.generate_answer(user_question, context=past_conversations + context)
    
    # Process the result to improve clarity, accuracy, and conciseness
    processed_answer = process_answer(generated_answer)

    # Save the current interaction to MongoDB
    save_user_session(user_id, user_question, processed_answer, context)
    
    return processed_answer

# Initialize the advanced text generation model
advanced_model = AdvancedTextGenerationModel()

# Streamlit UI
st.title("Herbal Remedy Chatbot")

user_id = st.sidebar.text_input("User ID", value="user123")
conversation_count = sessions_collection.count_documents({"user_id": user_id})

options = ["Start a new conversation"] + [f"Continue conversation {i+1}" for i in range(conversation_count)]

option = st.sidebar.radio("Options", options)

if option == "Start a new conversation":
    user_question = st.text_input("Ask your question:")
    if st.button("Submit"):
        if user_question:
            result = chatbot_interaction(user_id, user_question)
            st.write(f"Bot: {result}")
            # Ask a relevant follow-up question
            follow_up_question = "Would you like instructions on how to prepare the remedy?"
            st.write(f"Bot: {follow_up_question}")
else:
    conv_num = int(option.split()[-1])
    if conv_num <= conversation_count:
        sessions = list(sessions_collection.find({"user_id": user_id}).skip(conv_num-1).limit(1))
        if sessions:
            session = sessions[0]
            st.write(f"Continuing conversation {conv_num}")
            st.write(f"Question: {session['question']}")
            st.write(f"Answer: {session['answer']}")
            st.write(f"Context: {session['context']}")
            user_question = st.text_input("Ask a follow-up question:")
            if st.button("Submit"):
                if user_question:
                    result = chatbot_interaction(user_id, user_question)
                    st.write(f"Bot: {result}")
                    # Continue asking relevant questions based on conversation
                    follow_up_question = "Would you like to know more about this remedy?"
                    st.write(f"Bot: {follow_up_question}")
    else:
        st.write("Invalid conversation number. Please choose a valid conversation.")
