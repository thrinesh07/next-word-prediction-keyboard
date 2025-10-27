import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import types
import tensorflow.keras.preprocessing.text as new_module

# Flask app setup
app = Flask(__name__, static_folder="static", template_folder="templates")

# === Load Model & Tokenizer ===
MODEL_FILE = "Next_word.h5"
TOKENIZER_FILE = "tokenizer.pkl"
INDEX_TO_WORD_FILE = "index_to_word.pkl"

# Load Keras model
model = load_model(MODEL_FILE)

# Fix pickle import for modern Keras
sys.modules['keras.preprocessing.text'] = types.SimpleNamespace(
    Tokenizer=new_module.Tokenizer
)

# Load tokenizer
with open(TOKENIZER_FILE, "rb") as f:
    tokenizer = pickle.load(f)

# Load index_to_word mapping
with open(INDEX_TO_WORD_FILE, "rb") as f:
    index_to_word = pickle.load(f)

# Max sequence length for padding
max_seq_len = model.input_shape[1]

# === Predict next words ===
def predict_next_words(seed_text, top_k=3, temperature=1.0):
    seq = tokenizer.texts_to_sequences([seed_text])[0]
    seq = pad_sequences([seq], maxlen=max_seq_len, truncating="pre")
    
    preds = model.predict(seq, verbose=0)[0]
    preds = np.log(preds + 1e-12) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    
    top_indices = probs.argsort()[-top_k:][::-1]
    suggestions = [(index_to_word.get(i, ""), float(probs[i])) for i in top_indices]
    return suggestions

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    data = request.get_json()
    seed_text = data.get("seed_text", "").strip()
    if not seed_text:
        return jsonify({"suggestions": []})
    
    temperature = float(data.get("temperature", 1.0))
    suggestions = predict_next_words(seed_text, top_k=3, temperature=temperature)
    return jsonify({"suggestions": suggestions})

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
