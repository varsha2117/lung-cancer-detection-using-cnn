# 🫁 CT Lung Cancer Detection

A machine learning application that analyzes chest CT scans to estimate the probability of lung cancer using a compact Convolutional Neural Network (CNN).

## 🚀 Features

- **Medical Image Analysis**: Supports PNG, JPG, and DICOM file formats
- **Real-time Prediction**: Instant cancer probability estimation
- **User-friendly Interface**: Clean Streamlit web interface
- **Robust Preprocessing**: Handles various medical imaging formats
- **Production Ready**: Docker containerization and cloud deployment support

## 🏗️ Architecture

### Model Architecture
- **CNN Layers**: 2 convolutional layers (16→32 channels)
- **Pooling**: Max pooling for dimensionality reduction
- **Regularization**: 25% dropout to prevent overfitting
- **Classification**: Binary classification with sigmoid activation
- **Input Size**: 224×224 grayscale images

### Preprocessing Pipeline
- Image resizing to 224×224 pixels
- Grayscale conversion
- Intensity normalization (mean=0.5, std=0.5)
- DICOM file support with proper rescaling

## 📦 Installation

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd ct-lung-cancer-detector

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m streamlit run app.py
```

### Docker Deployment
```bash
# Build Docker image
docker build -t ct-lung-cancer-detector .

# Run container
docker run -p 8501:8501 ct-lung-cancer-detector
```

### Docker Compose
```bash
# Start with docker-compose
docker-compose up -d
```

## 🌐 Usage

1. **Access the Application**: Open http://localhost:8501 in your browser
2. **Upload CT Scan**: Choose a PNG, JPG, or DICOM file
3. **View Results**: Get instant cancer probability percentage
4. **Model Options**: Optionally specify custom model weights

## 📁 Project Structure

```
ct-lung-cancer-detector/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── Procfile             # Heroku deployment
├── DEPLOYMENT.md        # Deployment guide
├── utils/
│   ├── __init__.py
│   ├── model.py         # CNN model definition
│   └── preprocess.py    # Image preprocessing
└── .streamlit/
    └── config.toml      # Streamlit configuration
```

## 🔧 Dependencies

- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computing
- **PyDICOM**: Medical image format support

## 🚀 Deployment Options

### Cloud Platforms
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use provided Procfile
- **Railway**: Auto-detects Python applications
- **Render**: Web service deployment

### Self-hosted
- **Docker**: Containerized deployment
- **Local**: Direct Python execution

## ⚠️ Medical Disclaimer

**This tool is not a medical device and is not intended for diagnosis or treatment. Consult qualified medical professionals for clinical decisions.**

## 📊 Model Performance

- **Architecture**: Compact CNN optimized for medical imaging
- **Input**: 224×224 grayscale CT slices
- **Output**: Cancer probability (0-100%)
- **Processing**: CPU-optimized for accessibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical device regulations in your jurisdiction.

## 🔗 Links

- **Live Demo**: [Deploy to Streamlit Cloud]
- **Documentation**: See DEPLOYMENT.md for detailed setup
- **Issues**: Report bugs and feature requests
