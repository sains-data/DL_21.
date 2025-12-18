# Streamlit Cloud Deployment README

## Prerequisites
- Model file: `best_lstm_final_balanced.h5`
- Tokenizer file: `tokenizer_for_model_terbaik.pickle`
- All packages in `requirements.txt`

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository and `code_streamlit.py` as the main script
5. Click "Deploy"

**Note:** TensorFlow has been removed from requirements.txt to avoid build failures. The app works in two modes:

- **Production Mode (with TensorFlow):** Model loads and makes real predictions
- **Demo Mode (without TensorFlow):** Shows demo interface with fallback predictions

Streamlit Cloud will run in **Demo Mode** by default. For real model inference, you can:
- Add TensorFlow back locally by running: `pip install tensorflow>=2.13.0`
- Or install: `pip install -r requirements.txt` then manually `pip install tensorflow`

## Environment Variables (if needed)
Set these in Streamlit Cloud settings:
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_PORT=8501`

## Local Testing

To test locally **with model predictions:**

```bash
# Install full requirements including TensorFlow
pip install -r requirements.txt
pip install tensorflow>=2.13.0

# Run the app
streamlit run code_streamlit.py
```

To test locally **in demo mode (no TensorFlow):**

```bash
pip install -r requirements.txt
streamlit run code_streamlit.py
```

The app will be available at `http://localhost:8501`

## Troubleshooting

### Model not found
- Ensure `best_lstm_final_balanced.h5` is in the project root
- Check file paths in `MODEL_SEARCH_PATHS`

### Tokenizer not found
- Ensure `tokenizer_for_model_terbaik.pickle` is in the project root
- Check file paths in `tokenizer_init_paths`

### App shows demo mode instead of predictions
- TensorFlow is not installed (expected on Streamlit Cloud)
- Install TensorFlow locally to see real predictions
- This is by design to allow deployment in resource-constrained environments

### Installer returned non-zero exit code
- This usually means a dependency conflict during build
- TensorFlow has been removed to avoid this
- If the error persists, check individual package versions

## File Structure

```
/
├── code_streamlit.py              # Main Streamlit app
├── best_lstm_final_balanced.h5   # Trained model
├── tokenizer_for_model_terbaik.pickle # Tokenizer
├── requirements.txt               # Python dependencies (without TensorFlow)
├── runtime.txt                    # Python version
├── .streamlit/
│   ├── config.toml               # Streamlit configuration
│   └── secrets.toml              # Secrets (not committed)
└── README.md                      # This file
```
