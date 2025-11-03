# train_rnn.py
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

# ensure nltk resources
nltk.download('stopwords', quiet=True)
EN_STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Basic cleaning: remove HTML tags, non-letters, lowercase
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = text.lower()
    # optional: remove short words
    tokens = [w for w in text.split() if len(w) > 1 and w not in EN_STOPWORDS]
    return " ".join(tokens)

def load_data(csv_path, sample=None):
    df = pd.read_csv(csv_path)
    # Expect columns 'review' and 'sentiment'
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns")
    if sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    tqdm.pandas(desc="Cleaning text")
    df['clean_review'] = df['review'].progress_apply(clean_text)
    return df['clean_review'].tolist(), df['sentiment'].tolist()

def build_tokenizer(texts, num_words=20000):
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok

def build_model(vocab_size, embedding_dim=100, max_len=200, rnn_units=128, dropout=0.3):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(rnn_units, return_sequences=False)))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    print("Loading data from:", args.csv)
    texts, labels = load_data(args.csv, sample=args.sample)
    le = LabelEncoder()
    y = le.fit_transform(labels)  # negative->0, positive->1

    # Tokenize
    tokenizer = build_tokenizer(texts, num_words=args.vocab_size)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=args.max_len, padding='post', truncating='post')

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, random_state=42, stratify=y)

    print("Shapes:", X_train.shape, X_val.shape)
    vocab_size = min(args.vocab_size, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size, embedding_dim=args.embedding_dim, max_len=args.max_len, rnn_units=args.rnn_units, dropout=args.dropout)
    model.summary()

    # callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.save_dir + '/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    history = model.fit(X_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks)

    # evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation loss: {loss:.4f}  acc: {acc:.4f}")

    # save tokenizer and model
    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, 'final_model.keras'))
    import json
    with open(os.path.join(args.save_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(tokenizer.to_json())
    print("Saved model and tokenizer to", args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/IMDB Dataset.csv', help='path to csv')
    parser.add_argument('--save_dir', type=str, default='saved_model', help='where to save model and tokenizer')
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--rnn_units', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--sample', type=int, default=None, help='optionally use a subset for quick runs')
    args = parser.parse_args()
    main(args)