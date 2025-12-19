#!/usr/bin/env python3
"""
Export model parameters untuk digunakan di streamlit.
Script ini membaca tokenizer, kemudian export parameters ke pickle dan JSON.
"""

import pickle
import json
import os
from pathlib import Path

print("=" * 80)
print("[EXPORT] Mengekstrak model parameters...")
print("=" * 80)

# Paths
MODEL_PATH = "best_lstm_final_balanced.h5"
TOKENIZER_PATH = "tokenizer_for_model_terbaik.pickle"
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

# Load tokenizer
print(f"\n[1] Loading tokenizer dari {TOKENIZER_PATH}...")
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✓ Tokenizer loaded successfully")
    vocab_size = len(tokenizer.word_index) + 1
    print(f"  Vocab size: {vocab_size}")
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")
    exit(1)

# Verify model exists
print(f"\n[2] Verifying model {MODEL_PATH}...")
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✓ Model exists: {file_size:.1f} MB")
else:
    print(f"✗ Model not found: {MODEL_PATH}")
    exit(1)


# Extract parameters
print(f"\n[3] Extracting model parameters...")

params = {
    "vocab_size": vocab_size,
    "maxlen": 300,
    "embedding_dim": 100,
    "threshold": 0.40,  # Optimal threshold untuk balance precision-recall
    "model_type": "CNN-BiLSTM",
    "architecture": {
        "embedding_dim": 100,
        "conv_filters": 128,
        "conv_kernel_size": 5,
        "lstm_units": 64,
        "lstm_dropout": 0.3,
        "dense_units": 64,
        "dense_dropout": 0.3,
        "max_sequence_length": 300
    },
    "training_info": {
        "balanced_data": True,
        "class_weights": {0: 0.7, 1: 1.75},
        "loss_function": "focal_loss",
        "optimizer": "adam",
        "epochs_trained": 7,
        "early_stopping": True
    },
    "performance_metrics": {
        "accuracy": 0.8402,
        "precision": 0.7318,
        "recall": 0.6958,
        "f1_score": 0.7134,
        "roc_auc": 0.8676
    },
    "preprocessing": {
        "clean_text": True,
        "remove_stopwords": True,
        "tokenization": "keras_tokenizer",
        "padding_type": "post",
        "truncating": "post"
    }
}

print(f"✓ Parameters extracted:")
print(f"  - vocab_size: {params['vocab_size']}")
print(f"  - maxlen: {params['maxlen']}")
print(f"  - threshold: {params['threshold']}")
print(f"  - model_type: {params['model_type']}")

# Export to pickle
print(f"\n[4] Exporting to pickle and JSON...")
pickle_path = EXPORT_DIR / "model_params.pickle"
try:
    with open(pickle_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"✓ Saved to {pickle_path}")
except Exception as e:
    print(f"✗ Error saving pickle: {e}")

# Export to JSON
json_path = EXPORT_DIR / "model_params.json"
try:
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"✓ Saved to {json_path}")
except Exception as e:
    print(f"✗ Error saving JSON: {e}")

# Copy files to root (untuk compatibility)
print(f"\n[5] Copying tokenizer to root for compatibility...")
try:
    import shutil
    if os.path.exists(TOKENIZER_PATH):
        # Already at root
        print(f"✓ Tokenizer already at root: {TOKENIZER_PATH}")
    else:
        print(f"  Tokenizer path: {TOKENIZER_PATH}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("✓ MODEL PARAMETERS EXPORT COMPLETE")
print("=" * 80)
print(f"\nFiles created:")
print(f"  - {pickle_path}")
print(f"  - {json_path}")
print(f"\nThreshold untuk prediksi: {params['threshold']}")
print(f"Logic: prob < {params['threshold']} = BULLY, prob >= {params['threshold']} = NOT BULLY")
print("=" * 80)
