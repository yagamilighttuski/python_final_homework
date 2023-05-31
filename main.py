# required modules
import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

# loading the files we made previously
intents = json.loads(open("intense.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")


def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bagw(sentence):
    # separate out words from the input sentence
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            # check whether the word
            # is present in the input as well
            if word == w:
                # as the list of words
                # created earlier.
                bag[i] = 1

    # return a numpy array
    return np.array(bag)


def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            # prints a random response
            result = random.choice(i["responses"])
            break
    return result

import tkinter as tk
from tkinter import scrolledtext, END
window = tk.Tk()
window.title("Chatbot")
chat_history = scrolledtext.ScrolledText(window, state="disabled")
chat_history.pack(fill="both", expand=True)
user_input = tk.Entry(window)
user_input.pack(fill="x")
def send_message():
    message = user_input.get()
    user_input.delete(0, END)
    chat_history.configure(state="normal")
    chat_history.insert(END, "You: " + message + "\n")
    # Add code to process user input and generate a response
    response = get_chatbot_response(message)
    chat_history.insert(END, "Chatbot: " + response + "\n")
    chat_history.configure(state="disabled")

from keras.models import load_model
import pickle

# Load the trained model and other required data
model = load_model('chatbotmodel.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def get_chatbot_response(message):
    # Preprocess the user input message
    # Convert the message into the input format required by the model
    # Perform any necessary text processing, tokenization, lemmatization, etc.

    # Example: Tokenize and lemmatize the user input message
    word_list = nltk.word_tokenize(message)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]

    # Create a bag-of-words representation for the message
    bag = [1 if word in word_list else 0 for word in words]

    # Make predictions using the trained model
    input_data = np.array([bag])
    result = model.predict(input_data)[0]
    predicted_class_index = np.argmax(result)
    predicted_class = classes[predicted_class_index]

    # Return the appropriate response based on the predicted class
    # You can use the intents from your "intense.json" file to map the predicted class to a response
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            return response

    # If no appropriate response is found, return a default response
    return "I'm sorry, but I don't understand."

user_input.bind("<Return>", lambda event: send_message())
window.mainloop()

print("Chatbot is up!")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

