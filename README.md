# 🎬 IMDB Movie Review Sentiment Analysis

A deep learning-powered web application that analyzes movie reviews and predicts whether they are positive or negative using a Bidirectional LSTM neural network trained on the IMDB dataset.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [GPU Support](#gpu-support)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **🤖 Deep Learning Model**: Bidirectional LSTM architecture for accurate sentiment classification
- **📊 Real-time Predictions**: Instant sentiment analysis with confidence scores
- **💾 Model Persistence**: Train once, use forever - model is saved automatically
- **🖥️ GPU Acceleration**: Automatic GPU detection and utilization
- **🎯 High Accuracy**: Achieves 85-88% accuracy on test data
- **🎨 User-Friendly Interface**: Clean and intuitive Streamlit web interface
- **📝 Example Reviews**: Pre-loaded examples to test the model instantly
- **📈 Training Visualization**: Real-time training progress and accuracy plots

## 🎥 Demo

### Prediction Interface
```
Enter a movie review → Get instant sentiment analysis
✅ Positive Review (Confidence: 94.5%)
❌ Negative Review (Confidence: 87.2%)
```

### Training Interface
```
Click "Start Training" → Model trains in 3-5 minutes → Save & Use
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for faster training

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## 💻 Usage

### Running the Application

1. **Start the Streamlit app:**
```bash
streamlit run app.py
```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

3. **First Time Setup:**
   - Click "Start Training" to train the model (one-time, 3-5 minutes)
   - Model is automatically saved as `imdb_sentiment_model.h5`

4. **Making Predictions:**
   - Enter any movie review in the text area
   - Click "Analyze Sentiment"
   - Get instant results with confidence scores

### Command Line Options
```bash
# Run on specific port
streamlit run app.py --server.port 8080

# Run without browser auto-open
streamlit run app.py --server.headless true
```

## 🧠 Model Architecture

```
Input (Text Review)
    ↓
Embedding Layer (10,000 words, 128 dimensions)
    ↓
Bidirectional LSTM (64 units, return sequences)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dense Layer (64 units, ReLU activation)
    ↓
Dropout (50%)
    ↓
Output Layer (1 unit, Sigmoid activation)
    ↓
Sentiment (Positive/Negative)
```

### Model Parameters
- **Vocabulary Size**: 10,000 most frequent words
- **Maximum Sequence Length**: 250 words
- **Embedding Dimension**: 128
- **Training Dataset**: 25,000 IMDB reviews
- **Validation Split**: 20%
- **Batch Size**: 128
- **Epochs**: 3
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

### Performance Metrics
- **Test Accuracy**: 85-88%
- **Training Time**: 3-5 minutes (with GPU)
- **Inference Time**: <1 second per review

## 🖥️ GPU Support

### Automatic GPU Detection
The application automatically detects and utilizes available GPUs. GPU status is displayed in the sidebar.

### Setting Up GPU Support

#### Windows
```bash
# Install CUDA 11.8
Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive

# Install cuDNN 8.6
Download from: https://developer.nvidia.com/cudnn

# Verify
nvidia-smi
```

#### Linux
```bash
# Using conda (recommended)
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6

# Or using apt (Ubuntu/Debian)
sudo apt-get install cuda-11-8
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
```

#### Verify GPU
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### GPU Troubleshooting
If GPU is not detected:
1. Install NVIDIA drivers: [Download Here](https://www.nvidia.com/download/index.aspx)
2. Install CUDA Toolkit 11.8
3. Install cuDNN 8.6
4. Restart your computer
5. Verify installation using commands above

## 📁 Project Structure

```
imdb-sentiment-analysis/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── imdb_sentiment_model.h5        # Trained model (generated after training)
├── word_index.pkl                 # Word index mapping (generated after training)
│
└── .gitignore                     # Git ignore file
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'tensorflow'
```bash
pip install tensorflow==2.13.0
```

#### 2. GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall TensorFlow GPU
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

#### 3. Out of Memory Error
Reduce batch size in `app.py`:
```python
# Change from
batch_size=128

# To
batch_size=64
```

#### 4. Model Training Takes Too Long
- Ensure GPU is detected and working
- Close other applications to free up resources
- Consider training on a machine with better hardware

#### 5. Streamlit Connection Error
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Run again
streamlit run app.py
```

### Getting Help
- Check the [Issues](https://github.com/yourusername/imdb-sentiment-analysis/issues) page
- Open a new issue with detailed error messages
- Include system information: OS, Python version, TensorFlow version

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add comments for complex logic
- Update README if adding new features
- Test thoroughly before submitting PR

## 📊 Dataset Information

**IMDB Dataset**
- **Source**: TensorFlow Datasets / Keras
- **Size**: 50,000 movie reviews
- **Split**: 25,000 training, 25,000 testing
- **Classes**: Binary (Positive/Negative)
- **Language**: English
- **License**: Public Domain

## 🎯 Future Enhancements

- [ ] Multi-language support
- [ ] Batch prediction from CSV files
- [ ] API endpoint for predictions
- [ ] Model comparison (different architectures)
- [ ] Export predictions to file
- [ ] Advanced analytics dashboard
- [ ] Mobile-responsive UI improvements
- [ ] Docker containerization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the web application framework
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) for the movie reviews
- [Keras](https://keras.io/) for the high-level neural network API

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/imdb-sentiment-analysis](https://github.com/yourusername/imdb-sentiment-analysis)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

