
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

model = load_model('next_words.keras')
tokenizer = pickle.load(open('token.pkl', 'rb'))

def predict_word(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=model.input_shape[1])
    preds = np.argmax(model.predict(sequence), axis=-1)
    predicted_word = ""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    print(predicted_word)
    return predicted_word
while True:
    text=input("enter text")
    if text=="1":
        break
    else:
        text=text.split(" ")
        text=text[-3:]
        predict_word(model, tokenizer, text)
