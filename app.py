import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s) detected")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️ No GPU detected, running on CPU")

# Set page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# Constants
MAX_WORDS = 10000
MAX_LEN = 250
EMBEDDING_DIM = 128
MODEL_PATH = "imdb_sentiment_model.h5"
WORD_INDEX_PATH = "word_index.pkl"

@st.cache_resource
def load_word_index():
    """Load or create IMDB word index"""
    if os.path.exists(WORD_INDEX_PATH):
        with open(WORD_INDEX_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        word_index = imdb.get_word_index()
        with open(WORD_INDEX_PATH, 'wb') as f:
            pickle.dump(word_index, f)
        return word_index

@st.cache_resource
def load_model():
    """Load the trained model"""
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        return model
    return None

def train_and_save_model():
    """Train and save the sentiment analysis model"""
    # Load IMDB dataset
    st.info("📥 Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
    
    # Pad sequences
    st.info("🔄 Preprocessing data...")
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    
    # Build model
    st.info("🏗️ Building model architecture...")
    model = keras.Sequential([
        keras.layers.Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    st.text("Model Architecture:")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.code('\n'.join(model_summary))
    
    # Train model
    st.info("🚀 Training model... This will take 3-5 minutes.")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / 3)
            status_text.text(f"Epoch {epoch + 1}/3 - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")
    
    history = model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.2,
        callbacks=[StreamlitCallback()],
        verbose=0
    )
    
    # Evaluate
    st.info("📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Save model
    st.info("💾 Saving model...")
    model.save(MODEL_PATH)
    
    return model, history, test_acc

def preprocess_text(text, word_index):
    """Preprocess text for prediction"""
    words = text.lower().split()
    sequence = []
    for word in words:
        if word in word_index:
            index = word_index[word]
            if index < MAX_WORDS:
                sequence.append(index)
    
    padded = pad_sequences([sequence], maxlen=MAX_LEN)
    return padded

def predict_sentiment(model, text, word_index):
    """Predict sentiment of given text"""
    processed = preprocess_text(text, word_index)
    prediction = model.predict(processed, verbose=0)[0][0]
    return prediction

# Main app
def main():
    st.title("🎬 IMDB Movie Review Sentiment Analysis")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "This app uses a Bidirectional LSTM neural network trained on the IMDB dataset "
            "to predict whether a movie review is positive or negative."
        )
        
        # GPU Status
        st.markdown("### 🖥️ System Info")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            st.success(f"✅ GPU Detected: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                st.text(f"GPU {i}: {gpu.name}")
        else:
            st.warning("⚠️ Running on CPU")
        
        st.markdown("### Model Info")
        
        # Check model status
        if os.path.exists(MODEL_PATH):
            st.success("✅ Model: Trained & Ready")
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.text(f"Model Size: {file_size:.2f} MB")
        else:
            st.warning("⚠️ Model: Not Trained")
        
        st.markdown("### Architecture")
        st.text("• Embedding Layer\n• Bidirectional LSTM\n• Dense Layers\n• Dropout\n• Sigmoid Output")
        
        st.markdown("---")
        if st.button("🗑️ Delete Model", use_container_width=True):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
                st.success("Model deleted!")
                st.rerun()
    
    # Load word index
    word_index = load_word_index()
    
    # Initialize model in session state if not present
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Try to load model from file if session state is empty
    if st.session_state.model is None:
        loaded_model = load_model()
        if loaded_model is not None:
            st.session_state.model = loaded_model
    
    # Check if we have a model (either in session state or just loaded)
    if st.session_state.model is None:
        # Training interface
        st.warning("⚠️ No trained model found. Please train the model first.")
        
        st.header("🧠 Train Model")
        st.info(
            "⏱️ **Training Time:** Approximately 3-5 minutes\n\n"
            "📊 **Dataset:** 25,000 IMDB movie reviews\n\n"
            "🎯 **Expected Accuracy:** 85-88%"
        )
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            try:
                model, history, test_acc = train_and_save_model()
                
                st.success(f"✅ Model trained and saved successfully!")
                st.balloons()
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Accuracy", f"{test_acc*100:.2f}%")
                with col2:
                    st.metric("Test Loss", f"{history.history['loss'][-1]:.4f}")
                
                # Plot training history
                st.subheader("📈 Training History")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.line_chart(history.history['accuracy'])
                    st.caption("Training Accuracy")
                
                with col2:
                    st.line_chart(history.history['val_accuracy'])
                    st.caption("Validation Accuracy")
                
                st.info("🔄 Refresh the page to start making predictions!")
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
    
    else:
        # ==========================================
        # PREDICTION INTERFACE - MAIN AREA
        # ==========================================
        st.header("🔮 Predict Movie Review Sentiment")
        
        # Get model from session state
        model = st.session_state.model
        
        # Text input
        review_text = st.text_area(
            "Enter your movie review:",
            height=150,
            placeholder="Type or paste a movie review here...\n\nExample: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
        
        # Prediction
        if predict_button:
            if review_text and len(review_text.strip()) > 0:
                with st.spinner("🤔 Analyzing sentiment..."):
                    prediction = predict_sentiment(model, review_text, word_index)
                    
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Main result
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if prediction > 0.5:
                            st.success("### 😊 POSITIVE REVIEW")
                            sentiment_emoji = "😊"
                            sentiment_color = "green"
                        else:
                            st.error("### 😞 NEGATIVE REVIEW")
                            sentiment_emoji = "😞"
                            sentiment_color = "red"
                        
                        confidence = prediction if prediction > 0.5 else (1 - prediction)
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    with col2:
                        st.markdown("#### Sentiment Distribution")
                        st.progress(float(prediction))
                        
                        col_a, col_b = st.columns(2)
                        col_a.markdown(f"**😞 Negative**")
                        col_a.markdown(f"### {(1-prediction)*100:.1f}%")
                        col_b.markdown(f"**😊 Positive**")
                        col_b.markdown(f"### {prediction*100:.1f}%")
                    
                    # Additional info
                    st.markdown("---")
                    with st.expander("ℹ️ See detailed analysis"):
                        st.write(f"**Raw Prediction Score:** {prediction:.6f}")
                        st.write(f"**Review Length:** {len(review_text)} characters, {len(review_text.split())} words")
                        st.write(f"**Sentiment:** {'Positive' if prediction > 0.5 else 'Negative'}")
                        
            else:
                st.warning("⚠️ Please enter a review to analyze.")
        
        # Example reviews section
        st.markdown("---")
        st.subheader("💡 Try These Examples")
        
        examples = {
            "Positive 😊": "This movie was absolutely fantastic! The acting was superb, the cinematography was breathtaking, and the plot kept me engaged from start to finish. I highly recommend it to everyone. A true masterpiece!",
            "Negative 😞": "What a terrible waste of time and money. The plot made absolutely no sense, the acting was wooden and unconvincing, and I couldn't wait for it to end. Do not watch this movie under any circumstances.",
            "Mixed 🤔": "The movie had some good moments with decent special effects, but the story was pretty weak and predictable. Some of the acting was good while other performances fell flat. It's okay if you have nothing else to watch."
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Positive Example", use_container_width=True):
                st.session_state.selected_example = examples["Positive 😊"]
                st.rerun()
        
        with col2:
            if st.button("📝 Negative Example", use_container_width=True):
                st.session_state.selected_example = examples["Negative 😞"]
                st.rerun()
        
        with col3:
            if st.button("📝 Mixed Example", use_container_width=True):
                st.session_state.selected_example = examples["Mixed 🤔"]
                st.rerun()
        
        # Display selected example
        if 'selected_example' in st.session_state:
            st.info(f"**Selected Example:**\n\n{st.session_state.selected_example}")
            if st.button("✨ Analyze This Example"):
                prediction = predict_sentiment(model, st.session_state.selected_example, word_index)
                sentiment = "POSITIVE 😊" if prediction > 0.5 else "NEGATIVE 😞"
                confidence = prediction if prediction > 0.5 else (1 - prediction)
                st.success(f"**Result:** {sentiment} (Confidence: {confidence*100:.1f}%)")

if __name__ == "__main__":
    main()