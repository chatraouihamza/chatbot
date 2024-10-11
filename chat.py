import numpy as np
import json
import tensorflow as tf
import random
import pickle
from util import tokenize, bag_of_words  # Ensure 'stem' function is not needed or update import if necessary
from tensorflow.keras.models import load_model


# Load necessary resources
model_path = 'my_model.h5'  # Path to your saved model
loaded_model = load_model(model_path)

# Load training data (words and classes)
with open("training_data.pkl", "rb") as f:
    data = pickle.load(f)
words = data['words']
classes = data['classes']

# Load intents from KB.json
with open('Data/KB.json') as json_data:
    intents = json.load(json_data)

# Preprocess input for model
def preprocess_input(input_text, all_words):
    tokenized_text = tokenize(input_text)  # Tokenize the sentence
    bag = bag_of_words(tokenized_text, all_words)  # Convert to bag of words
    return np.array(bag).reshape(1, -1)  # Reshape to match model's expected input shape

def classify(sentence):
    # Tokenize and convert sentence to bag of words
    tokenized_sentence = tokenize(sentence)
    bow_array = bag_of_words(tokenized_sentence, words)
    
    # Print the BOW array to ensure it's correct
    # print(f"BOW array: {bow_array}")
    
    # Predict using the model
    prediction = loaded_model.predict(np.array([bow_array]))
    results = prediction[0]
    
    # Print the raw prediction results
    # print(f"Raw model prediction: {results}")
    
    # Sort the results by strength of probability (highest first)
    sorted_results = [(i, r) for i, r in enumerate(results)]
    sorted_results.sort(key=lambda x: x[1], reverse=True)
    
    # Print sorted results for debugging
    # print(f"Sorted results (all predictions): {sorted_results}")
    
    return [(classes[r[0]], r[1]) for r in sorted_results]
# Get response from KB.json based on classification
def get_response(sentence):
    results = classify(sentence)
    
    # Check if results are empty
    if not results:
        # print("No result passed the threshold.")
        return "Sorry, I didn't understand that."
    
    # print(f"Top classification result: {results[0]}")
    
    # Look for the intent matching the top result
    for intent in intents['intents']:
        if intent['tag'] == results[0][0]:  # Match intent with classified tag
            return random.choice(intent['responses'])
    
    return "Sorry, I didn't understand that."
# Main chat function
def chat():
    print("Chatbot is running. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Exiting chat...")
            break
        
        response = get_response(user_input)
        print(f"Bot: {response}")

# Ensure resources are loaded before starting the chat
if __name__ == "__main__":
    chat()



