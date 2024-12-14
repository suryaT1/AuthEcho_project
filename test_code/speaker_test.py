import gradio as gr
import numpy as np
import librosa
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Paths to your models and label encoders
lstm_speaker_model = '/content/lstm_speaker_model.h5'
lstm_gender_model = '/content/lstm_gender_model.h5'
lstm_speaker_label = '/content/lstm_speaker_label.pkl'
lstm_gender_label = '/content/lstm_gender_label.pkl'

# ------------------- Feature Extraction -------------------
def extract_features(audio_data, max_len=34):
    """Extract MFCC features from an audio file."""
    audio, sr = librosa.load(audio_data, sr=None)

    # Extract MFCC features (13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Spectral Features: Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Features: Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Combine only a subset of features (to match the model's expected input size)
    features = np.hstack([mfccs_mean[:13], chroma_mean[:13], spectral_contrast_mean[:8]])

    # Padding or truncating to fixed length (max_len)
    if features.shape[0] < max_len:
        padding = np.zeros((max_len - features.shape[0],))
        features = np.concatenate((features, padding))
    elif features.shape[0] > max_len:
        features = features[:max_len]

    return features

def preprocess_audio_for_model(audio_data, max_len=34):
    """Preprocess audio file for model prediction."""
    features = extract_features(audio_data, max_len=max_len)
    features = features.reshape(1, 1, features.shape[0])  # Shape for LSTM: (samples, timesteps, features)
    return features

# ------------------- Load Pre-trained Models and Label Encoders -------------------
def load_trained_model(model_path='/content/lstm_speaker_model.h5'):
    """Load the trained speaker model."""
    return tf.keras.models.load_model(model_path)

def load_gender_model(model_path='/content/lstm_gender_model.h5'):
    """Load the trained gender model."""
    return tf.keras.models.load_model(model_path)

def load_label_encoder(label_encoder_path='/content/lstm_speaker_label.pkl'):
    """Load the label encoder for speaker labels."""
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

def load_gender_label_encoder(label_encoder_path='/content/lstm_gender_label.pkl'):
    """Load the label encoder for gender labels."""
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

# ------------------- Predict Top 3 Speakers and Gender -------------------
def predict_top_3_speakers_and_gender(audio_data, speaker_model, gender_model, speaker_encoder, gender_encoder, max_len=34, threshold=50):
    """Predict the top 3 speakers and gender from an uploaded audio file."""
    features = preprocess_audio_for_model(audio_data, max_len=max_len)

    # Predict the speaker probabilities
    speaker_pred = speaker_model.predict(features)

    # Get the top speaker prediction confidence
    top_confidence = np.max(speaker_pred[0]) * 100  # Convert to percentage
    if top_confidence < threshold:
        return ["Unknown"], [top_confidence], "Unknown"

    # Get top 3 speakers
    top_3_speakers_idx = np.argsort(speaker_pred[0])[::-1][:3]
    top_3_speakers_probs = speaker_pred[0][top_3_speakers_idx] * 100  # Convert to percentages
    top_3_speakers = speaker_encoder.inverse_transform(top_3_speakers_idx)

    # Predict the gender
    gender_pred = gender_model.predict(features)  # Gender model needs 1D features
    predicted_gender = gender_encoder.inverse_transform([np.argmax(gender_pred)])[0]

    return top_3_speakers, top_3_speakers_probs, predicted_gender

# ------------------- Gradio Interface -------------------
def gradio_interface(audio):
    # Load the trained models and label encoders
    speaker_model = load_trained_model(lstm_speaker_model)  # Speaker model
    gender_model = load_gender_model(lstm_gender_model)  # Gender model
    speaker_encoder = load_label_encoder(lstm_speaker_label)  # Speaker label encoder
    gender_encoder = load_gender_label_encoder(lstm_gender_label)  # Gender label encoder

    # Predict the top 3 speakers and gender from the uploaded audio file
    top_3_speakers, top_3_speakers_probs, predicted_gender = predict_top_3_speakers_and_gender(
        audio, speaker_model, gender_model, speaker_encoder, gender_encoder, threshold=50
    )

    # Return results as a formatted string for Gradio output
    result = f"The top 3 predicted speakers are:\n"
    for speaker, prob in zip(top_3_speakers, top_3_speakers_probs):
        result += f"{speaker}: {prob:.2f}%\n"

    result += f"\nThe predicted gender is: {predicted_gender}"

    return result

# Gradio interface creation
demo = gr.Interface(
    fn=gradio_interface,  # The function to predict speaker and gender
    inputs=gr.Audio(type="filepath"),  # Audio input (file upload)
    outputs="text",  # Output the prediction result as text
    live=False,  # Disable live feedback
    title="Speaker and Gender Prediction",
    description="Upload or record an audio file to predict the top 3 speakers and gender.",
    allow_flagging="never",  # Disable flagging
    theme="compact",  # Set the theme
    css="""
    body {
        margin: 0;
        padding: 0;
        background-color: #f1f1f1;
        font-family: 'Roboto', sans-serif;
    }

    .gradio-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    h1, p {
        color: #333;
    }
    """
)

# Launch Gradio app
demo.launch()
