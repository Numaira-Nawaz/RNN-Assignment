# ===============================================================
# 🎬 Sentiment Analysis on IMDB Dataset using BiLSTM
# Author: ChatGPT (Numaira version)
# ===============================================================

import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import json

# Ensure nltk resources
nltk.download('stopwords', quiet=True)
EN_STOPWORDS = set(stopwords.words('english'))

# ===============================================================
# 🧹 Clean Text
# ===============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)                     # Remove HTML tags
    text = re.sub(r"[^a-zA-Z']", " ", text)                # Keep letters
    text = text.lower()
    words = text.split()

    # Handle negations
    negations = {"not", "no", "never", "n't"}
    new_words = []
    skip_next = False

    for i, w in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if w in negations and i + 1 < len(words):
            new_words.append(w + "_" + words[i + 1])
            skip_next = True
        else:
            new_words.append(w)

    words = [w for w in new_words if w not in EN_STOPWORDS and len(w) > 1]
    return " ".join(words)

# ===============================================================
# 📂 Load Data
# ===============================================================
def load_data(csv_path, sample=None):
    # Fix CSV parser error
    df = pd.read_csv(csv_path, quoting=3, on_bad_lines='skip')

    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns")

    if sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)

    tqdm.pandas(desc="Cleaning text")
    df['clean_review'] = df['review'].progress_apply(clean_text)

    return df['clean_review'].tolist(), df['sentiment'].tolist()

# ===============================================================
# 🧠 Tokenizer + Model Builder
# ===============================================================
def build_tokenizer(texts, num_words=20000):
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok

def build_model(vocab_size, embedding_dim=100, max_len=200, rnn_units=128, dropout=0.3):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(rnn_units, return_sequences=False)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ===============================================================
# 🚀 Main Function
# ===============================================================
def main(args):
    print("📥 Loading data from:", args.csv)
    texts, labels = load_data(args.csv, sample=args.sample)

    le = LabelEncoder()
    y = le.fit_transform(labels)  # negative→0, positive→1

    # Tokenize and pad
    tokenizer = build_tokenizer(texts, num_words=args.vocab_size)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=args.max_len, padding='post', truncating='post')

    # Train / Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )

    print("✅ Shapes -> Train:", X_train.shape, "Val:", X_val.shape)
    vocab_size = min(args.vocab_size, len(tokenizer.word_index) + 1)

    model = build_model(
        vocab_size,
        embedding_dim=args.embedding_dim,
        max_len=args.max_len,
        rnn_units=args.rnn_units,
        dropout=args.dropout,
    )

    model.summary()

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.save_dir, "best_model.h5"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    # Fit model
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"\n📊 Validation loss: {loss:.4f}  |  acc: {acc:.4f}")

    # Save model + tokenizer
    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, "final_model.keras"))
    with open(os.path.join(args.save_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    print(f"\n✅ Model + Tokenizer saved in → {args.save_dir}")

# ===============================================================
# 🏁 Entry
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="IMDB Dataset.csv", help="path to IMDB CSV")
    parser.add_argument("--save_dir", type=str, default="saved_model", help="where to save model/tokenizer")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--rnn_units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=None, help="for quick tests (e.g. 2000)")
    args = parser.parse_args()
    main(args)
