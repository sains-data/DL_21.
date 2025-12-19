#!/usr/bin/env python3
"""
Script untuk load model dengan error handling yang lebih baik.
"""

import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)
print("Model file:", "best_lstm_final_balanced.h5")
print("File exists:", os.path.exists("best_lstm_final_balanced.h5"))
print("File size:", os.path.getsize("best_lstm_final_balanced.h5") / (1024*1024), "MB")

print("\nAttempting to load model...")

# Try different approaches
approaches = [
    ("Standard load", lambda: tf.keras.models.load_model('best_lstm_final_balanced.h5')),
    ("Load with custom_objects={}", lambda: tf.keras.models.load_model('best_lstm_final_balanced.h5', custom_objects={})),
    ("Load with compile=False", lambda: tf.keras.models.load_model('best_lstm_final_balanced.h5', compile=False)),
]

for name, loader in approaches:
    try:
        print(f"\n[{name}]")
        model = loader()
        print(f"✓ SUCCESS")
        print(f"  Model type: {type(model)}")
        print(f"  Model summary:")
        model.summary()
        break
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:150]}")
