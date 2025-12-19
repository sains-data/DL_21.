#!/usr/bin/env python3
"""
Test script untuk verifikasi prediksi cyberbullying.
Script ini test model tanpa streamlit.
"""

import pickle
import numpy as np
import os

print("=" * 80)
print("[TEST] Testing Cyberbullying Detection")
print("=" * 80)

# Load tokenizer
print("\n[1] Loading tokenizer...")
try:
    with open('tokenizer_for_model_terbaik.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✓ Tokenizer loaded")
    vocab_size = len(tokenizer.word_index) + 1
    print(f"  Vocab size: {vocab_size}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Load stopwords
print("\n[2.5] Loading stopwords...")
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('indonesian'))
    print(f"✓ Stopwords loaded: {len(stop_words)} words")
except Exception as e:
    print(f"✗ Error: {e}")
    stop_words = set()

# Load TensorFlow
print("\n[3] Loading TensorFlow...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} loaded")
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Load model
print("\n[3] Loading model...")
try:
    model = load_model('best_lstm_final_balanced.h5', compile=False)
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    print("  This may be due to TensorFlow version mismatch.")
    print("  Model can still be used if file exists.")
    model = None

if model is None:
    print("⚠️  Warning: Model load failed. Cannot run predictions.")
    exit(1)

# Load model parameters
print("\n[4] Loading model parameters...")
try:
    with open('exports/model_params.pickle', 'rb') as f:
        params = pickle.load(f)
    threshold = params.get('threshold', 0.46755287051200867)
    print(f"✓ Parameters loaded")
    print(f"  Threshold: {threshold}")
except Exception as e:
    print(f"⚠️  Error loading params: {e}")
    threshold = 0.46755287051200867
    print(f"  Using default threshold: {threshold}")

# Prediction function
def predict(text):
    """Predict cyberbullying"""
    import re
    import string
    
    # Clean text
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords (IMPORTANT: model was trained with this step)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([text])
    X = pad_sequences(seq, maxlen=300, padding='post', truncating='post')
    
    # Predict
    prob = float(model.predict(X, verbose=0).ravel()[0])
    
    # Decision: prob >= threshold = BULLY
    label = "BULLY" if prob >= threshold else "NOT BULLY"
    
    return label, prob, text

# Test cases
print("\n" + "=" * 80)
print("[TEST CASES]")
print("=" * 80)

test_cases = [
    ("kamu baik banget", "NOT BULLY"),  # Safe
    ("bodoh tolol", "BULLY"),  # Bully
    ("tolol", "BULLY"),  # Bully
    ("dongo", "BULLY"),  # Bully (slang bully)
    ("halo apa kabar", "NOT BULLY"),  # Safe
    ("kontol kampret", "BULLY"),  # Bully
    ("selamat datang", "NOT BULLY"),  # Safe
]

passed = 0
failed = 0

for test_text, expected in test_cases:
    label, prob, cleaned = predict(test_text)
    status = "✓" if label == expected else "✗"
    
    if label == expected:
        passed += 1
    else:
        failed += 1
    
    print(f"\n{status} Input: '{test_text}'")
    print(f"  Expected: {expected}")
    print(f"  Got:      {label} (prob: {prob:.4f}, thr: {threshold:.4f})")
    print(f"  Cleaned:  '{cleaned}'")

# Summary
print("\n" + "=" * 80)
print("[SUMMARY]")
print("=" * 80)
print(f"Total:  {len(test_cases)} tests")
print(f"Passed: {passed} tests")
print(f"Failed: {failed} tests")

if failed == 0:
    print("\n✓ ALL TESTS PASSED!")
else:
    print(f"\n✗ {failed} test(s) failed")

print("=" * 80)
