import gradio as gr
import pickle
import numpy as np
import librosa
from keras.models import load_model
from datetime import datetime
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the trained RNN model and scaler
model_rnn = load_model('/content/drive/MyDrive/CAP_GEN_PRED_V1/rnn_model.h5')  # Ensure the model is compatible with your Keras/TensorFlow version
print("RNN Model loaded!")

with open('/content/drive/MyDrive/CAP_GEN_PRED_V1/rnn_model.pkl', 'rb') as f:  # Load the scaler
    scaler = pickle.load(f)
print("Scaler loaded!")

# Define your Google Drive path
UPLOAD_FOLDER = '/content/drive/MyDrive/Uploads'  # The folder in your Google Drive where audio files will be saved

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Feature extraction function
def extract_features_from_audio(audio):
    print(f"Extracting features from the audio...")  # Debug line
    y, sr = librosa.load(audio, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Extract Pitch, Formants, Chroma, Spectral Contrast features
    pitch, _ = librosa.core.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0])
    pitch_std = np.std(pitch[pitch > 0])

    # Formant extraction using pyformants library
    sound = parselmouth.Sound(audio_file_path)
    formants = sound.to_formant_burg()  # Use Burg method for formant estimation

    # Get formant values at the midpoint of the sound
    midpoint = int(sound.duration / 2 * sound.sampling_frequency)
    formant_1 = formants.get_value_at_time(1, midpoint / sound.sampling_frequency)
    formant_2 = formants.get_value_at_time(2, midpoint / sound.sampling_frequency)
    formant_3 = formants.get_value_at_time(3, midpoint / sound.sampling_frequency)

    # Chroma and Spectral Contrast features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_contrast_std = np.std(spectral_contrast, axis=1)

    # Combine all features into a single array
    features = np.concatenate([mfcc_mean, mfcc_std,
                               [pitch_mean, pitch_std],
                               [formant_1, formant_2, formant_3],
                               chroma_mean, chroma_std,
                               spectral_contrast_mean, spectral_contrast_std])

    return features

# Gender prediction function
def predict_gender(audio):
    if audio is None:
        return "No audio provided!"

    # Extract features from the uploaded audio file
    features = extract_features_from_audio(audio)

    # Reshape the features to match model input shape (1, num_features)
    features = features.reshape(1, -1)

    # Standardize the features using the scaler
    features_scaled = scaler.transform(features)

    # Predict the gender using the RNN model
    prediction = model_rnn.predict(features_scaled)

    # Confidence for Male (class 0) and Female (class 1)
    male_confidence = prediction[0][0] * 100
    female_confidence = prediction[0][1] * 100

    # Determine the gender with the highest confidence
    predicted_gender = "Male" if male_confidence > female_confidence else "Female"

    # Return the result with confidence percentages
    return f"Predicted Gender: {predicted_gender}\nMale: {male_confidence:.2f}%\nFemale: {female_confidence:.2f}%"

# Create Gradio Interface
demo = gr.Interface(
    fn=predict_gender,  # The function to predict gender
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),  # Audio input (file or microphone)
    outputs="text",  # Output the prediction result as text
    live=False,  # Disable live feedback
    title="Speaker Gender Prediction",
    description="Upload or record an audio file to predict the speaker's gender.",
    allow_flagging="never",  # Disable flagging
    theme="compact",  # Set the theme
    css="""
        body {
            margin: 0;
            padding: 0;
            height: 100vh;
            background-image: url('https://media.datacenterdynamics.com/media/images/kyndryl_original_large.original.jpg');
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }

        .gradio-container {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
        }
    """  # Add custom CSS if needed
)

# Launch the interface
demo.launch()
