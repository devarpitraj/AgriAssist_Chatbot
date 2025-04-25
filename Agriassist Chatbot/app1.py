from flask import Flask, request, jsonify
import json
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from langdetect import detect
from googletrans import Translator

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the Sentence-BERT model (lightweight version)
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Translator for language translation
translator = Translator()

# Load the dataset from Bigdata.jsonl
def load_dataset(filepath):
    qa_pairs = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                qa_pairs.append((data["prompt"].lower(), data["completion"]))
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
    return qa_pairs

# Path to dataset
DATASET_PATH = "Bigdata_cleaned.jsonl"
qa_data = load_dataset(DATASET_PATH)

# Compute SBERT embeddings for all stored questions
if qa_data:
    questions = [q[0] for q in qa_data]
    question_embeddings = sbert_model.encode(questions, convert_to_tensor=True)

# Predefined responses
predefined_responses = {
    r"how are you": "I'm just a chatbot, but I'm doing great! How about you?",
    r"how's it going": "Everything is running smoothly! How can I assist you?",
    r"how do you do": "I'm doing well! Thanks for asking. How can I help?",
    r"what is your name": "I'm your Agricultural Chatbot! Here to assist you with farming-related queries.",
    r"who are you": "I'm an AI-powered agricultural chatbot, here to help farmers and agriculture enthusiasts.",
    r"what can you do": "I can provide information related to farming, crops, and agricultural best practices.",
    r"who created you": "I was developed as part of a research project on AI and ML-powered agricultural chatbots.",
    r"are you a robot": "Yes, I'm an AI chatbot, designed to help with agricultural questions.",
    r"do you speak other languages": "Currently, I only support English.",
    r"tell me a joke": "Why did the farmer win an award? Because he was outstanding in his field!",
    r"tell me something interesting": "Did you know? Earthworms can improve soil fertility by breaking down organic matter into nutrient-rich compost!",
    r"hello": "Hello! How can I assist you today?",
    r"hi": "Hi there! What would you like to know?",
    r"hey": "Hey! How’s it going?",
    r"good morning": "Good morning! Hope you have a great day ahead.",
    r"good afternoon": "Good afternoon! How can I help?",
    r"good evening": "Good evening! Need any assistance?",
    r"good night": "Good night! Take care and rest well.",
    r"howdy": "Howdy! What brings you here today?",
    r"yo": "Yo! What's up?",
    r"namaste": "Namaste! How can I assist you?",
    r"hola": "Hola! How can I help you?",
    r"bonjour": "Bonjour! What do you need assistance with?"
}

# Function to find the best matching response
def find_best_response(user_input):
    if not qa_data:
        return "Error: No data available.", 0.0

    original_input = user_input.strip()
    detected_lang = detect(original_input)

    # Translate Hindi to English if needed
    if detected_lang == 'hi':
        translated = translator.translate(original_input, src='hi', dest='en').text
    else:
        translated = original_input

    translated = translated.lower().strip()

    # Predefined responses
    for pattern, response in predefined_responses.items():
        if re.fullmatch(pattern, translated):
            if detected_lang == 'hi':
                response = translator.translate(response, src='en', dest='hi').text
            return response, 1.0

    # SBERT matching
    user_embedding = sbert_model.encode(translated, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]

    best_match_idx = int(np.argmax(similarities))
    best_match_score = float(similarities[best_match_idx])

    if best_match_score < 0.5:
        fallback = "I'm sorry, I don't have an answer for that."
        if detected_lang == 'hi':
            fallback = translator.translate(fallback, src='en', dest='hi').text
        return fallback, best_match_score

    response = qa_data[best_match_idx][1]
    if detected_lang == 'hi':
        response = translator.translate(response, src='en', dest='hi').text

    return response, best_match_score

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Agricultural Chatbot API!"})

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a valid question.", "confidence": 0.0}), 400
    
    bot_response, confidence_score = find_best_response(user_input)
    
    return jsonify({
        "response": bot_response,
        "confidence": round(confidence_score, 2)
    })

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
