import pandas as pd
import requests
import pickle
import tensorflow as tf
import random 
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text=requests.get(url).text.lower()
print("Total characters: ",len(text))


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text.split())  
total_words = len(tokenizer.word_index) + 1
print("Total unique words:", total_words)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


index_to_word = {index: word for word, index in tokenizer.word_index.items()}
with open("index_to_word.pkl", "wb") as f:
    pickle.dump(index_to_word, f)
    
from tensorflow.keras.preprocessing.sequence import pad_sequences
input_sequences=[]
for line in text.split('\n'):
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_seq=token_list[:i+1]
        input_sequences.append(n_gram_seq)

max_seq_len=max(len(seq) for seq in input_sequences)
input_sequences=pad_sequences(input_sequences,maxlen=max_seq_len,padding='pre')

print(input_sequences)

X,y=input_sequences[:,:-1],input_sequences[:,-1]
# X → all words except the last one (input).
# y → last word (label/output).
# to_categorical → one-hot encode labels for classification.
import tensorflow as tf
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print("Training smaples :",X.shape)

from keras.models import *
from keras.layers import *
model = Sequential([
    Embedding(total_words, 200, input_length=max_seq_len-1),   
    LSTM(256, return_sequences=True),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(total_words, activation="softmax")
])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(X, y, batch_size=64, epochs=100, verbose=1, callbacks=[early_stop])

loss,acc=model.evaluate(X,y)
print("Acc: ",acc)

model.save("Next_word.h5")
print("Model and tokenizer saved successfully!")


def predict_next_word(seed_text, next_words=100):
    """Generate next words for a given seed text."""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences(seed_text.split())
        token_list = [t[0] for t in token_list if t]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted_index = np.argmax(model.predict(token_list, verbose=0), axis=1)[0]
        output_word = index_to_word.get(predicted_index, "")
        seed_text += " " + output_word
    return seed_text

test_seed = "We are accounted"
generated_text = predict_next_word(test_seed, next_words=100)
print("Generated Text:\n", generated_text)