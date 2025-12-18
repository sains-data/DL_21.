#!/bin/bash
# Quick Start Guide for Streamlit Cloud Deployment
# Run this to verify everything before deploying

echo "=========================================="
echo "Streamlit Cyberbullying Detection System"
echo "Deployment Verification Script"
echo "=========================================="
echo ""

# Check Python version
echo "1. Python Version:"
python3 --version
echo ""

# Check required files
echo "2. Required Files:"
for file in code_streamlit.py requirements.txt runtime.txt best_lstm_final_balanced.h5 tokenizer_for_model_terbaik.pickle .streamlit/config.toml; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (MISSING)"
    fi
done
echo ""

# Check Git status
echo "3. Git Status:"
git log --oneline -1
echo ""

# Install dependencies
echo "4. Installing Dependencies..."
pip install -r requirements.txt -q
echo "   ✓ Dependencies installed"
echo ""

# Run Streamlit
echo "5. Starting Streamlit Application..."
echo "   App will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run code_streamlit.py
