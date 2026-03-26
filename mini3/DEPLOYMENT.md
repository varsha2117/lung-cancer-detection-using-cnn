# CT Lung Cancer Detection - Deployment Guide

## 🚀 Quick Deploy Options

### Option 1: Docker (Recommended)
```bash
# Build and run with Docker
docker build -t ct-lung-cancer-detector .
docker run -p 8501:8501 ct-lung-cancer-detector
```

### Option 2: Docker Compose
```bash
# Start with docker-compose
docker-compose up -d
```

### Option 3: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python -m streamlit run app.py
```

## ☁️ Cloud Deployment

### Heroku
1. Create `Procfile`:
   ```
   web: python -m streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy:
   ```bash
   git add .
   git commit -m "Deploy CT Lung Cancer Detector"
   git push heroku main
   ```

### Railway
1. Connect your GitHub repository
2. Railway will auto-detect Python and install dependencies
3. Set port: `8501`

### Render
1. Create new Web Service
2. Connect GitHub repository
3. Build command: `pip install -r requirements.txt`
4. Start command: `python -m streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## 🔧 Environment Variables
- `PORT`: Server port (default: 8501)
- `MODEL_WEIGHTS_PATH`: Path to model weights file

## 📁 Required Files
- `app.py` - Main application
- `requirements.txt` - Python dependencies
- `utils/` - Utility modules
- `model_weights.pth` - Optional trained model weights

## 🏥 Medical Disclaimer
This tool is not a medical device and is not intended for diagnosis or treatment. Consult qualified medical professionals for clinical decisions.
