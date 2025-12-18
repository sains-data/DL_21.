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

## Environment Variables (if needed)
Set these in Streamlit Cloud settings:
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_PORT=8501`

## Local Testing

```bash
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

### Memory issues
- The app uses demo fallback mode if TensorFlow is unavailable
- This is by design to allow deployment in resource-constrained environments

## File Structure

```
/
├── code_streamlit.py              # Main Streamlit app
├── best_lstm_final_balanced.h5   # Trained model
├── tokenizer_for_model_terbaik.pickle # Tokenizer
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version
├── .streamlit/
│   ├── config.toml               # Streamlit configuration
│   └── secrets.toml              # Secrets (not committed)
└── README.md                      # This file
```
