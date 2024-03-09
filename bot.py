# from keras.models import load_model
import re
import numpy as np
# import keras
# from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences
import telebot
from telebot.types import Message
import json
import pickle
import re
import string
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('wordnet')

# Replace with your Telegram Bot API token
API_TOKEN = 'Telegram Bot API token'
bot = telebot.TeleBot(API_TOKEN)

dir_model = 'model/'

# Fungsi untuk memuat model dari file
def load_model(model_filename, vectorizer_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(vectorizer_filename, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return model, vectorizer

# Fungsi untuk melakukan prediksi
def predict_answer(user_input, model, vectorizer):
    user_input_tfidf = vectorizer.transform([user_input])
    answer = model.predict(user_input_tfidf)[0]
    return answer

def preprocessing_text(text):
    # Mengganti karakter newline dengan spasi
    text = re.sub(r'\n', ' ', text)

    # Menghapus tanda kurung buka dan tutup
    text = re.sub(r'\(', '', text)
    text = re.sub(r'\)', '', text)

    # Menghapus koma
    text = re.sub(r',', '', text)

    # Menghapus tanda hubung
    text = re.sub(r'-', '', text)

    # Menghapus tanda slash
    text = re.sub(r'/', '', text)

    # Menghapus tanda tanya
    text = re.sub(r'\?', '', text)

    # Menghapus karakter khusus
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)

    # Menghapus URL
    text = re.sub(r'http\S+', '', text)

    # Mengubah teks menjadi huruf kecil (case folding)
    text = text.lower()

    # Tokenisasi menggunakan split
    words = text.split()  # Ini adalah tokenisasi sederhana, Anda dapat menggantinya dengan metode tokenisasi yang lebih canggih jika diperlukan

    # Menghapus stop words (menggunakan Sastrawi)
    stopword_factory = StopWordRemoverFactory()
    stop_words = stopword_factory.get_stop_words()  # Mengambil daftar stop words dalam bahasa Indonesia
    words = [word for word in words if word not in stop_words]

    # Stemming (menggunakan Sastrawi)
    stemmer = StemmerFactory().create_stemmer()
    text = ' '.join([stemmer.stem(word) for word in words])

    return text

# Memuat model dan vectorizer
classifier_model, tfidf_vectorizer = load_model(dir_model+'classifier_model.pkl', dir_model+'tfidf_vectorizer.pkl')

# Fungsi untuk mendapatkan jawaban berdasarkan input pengguna
def get_answer_using_model(user_input):
    # Lakukan preprocessing pada input pengguna
    preprocessed_input = preprocessing_text(user_input)
    # Lakukan prediksi jawaban
    answer = predict_answer(preprocessed_input, classifier_model, tfidf_vectorizer)
    return answer

# Load models
# encoder_model = load_model(dir_model+'encoder_model.h5')
# # # decoder_model = load_model(dir_model+'decoder_model.h5')

# # # # Load tokenizer from JSON file
# # # with open(dir_model+'tokenizer.json', 'r', encoding='utf-8') as f:
# # #     tokenizer_json = json.load(f)
# # #     tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# # # with open(dir_model+'jawabanLabel.json', 'r') as f:
# # #     reverse_label_dict = json.load(f)

# # max_encoder_seq_length = 27

# def predict(input_text):
#     input_text = preprocess_text(input_text)
#     input_sequence = tokenizer.texts_to_sequences([input_text])
#     padded_sequence = pad_sequences(input_sequence, maxlen=max_encoder_seq_length, padding='post')
#     predicted_output = decoder_model.predict(padded_sequence)
#     predicted_label_index = np.argmax(predicted_output)
#     predicted_label = reverse_label_dict[str(predicted_label_index)]
#     return predicted_label

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    bot_response = get_answer_using_model(user_input)
    bot.send_message(message.chat.id, bot_response)


bot.polling()