# AuthEcho_project
This Project contains the well trained deep learning models to predict the Speaker as well Gender


This repository contains a Speaker and Gender Prediction System built using TensorFlow, Librosa, and Gradio. The application predicts the top 3 speakers and their probabilities from an audio file and determines the speaker's gender. It can also classify unknown speakers by leveraging a confidence threshold.

Features
Predicts the top 3 speakers from an audio file.
Determines the gender of the speaker.
Handles unknown speakers by using a confidence threshold.
Provides an easy-to-use Gradio interface for testing the system.

Getting Started
Prerequisites
To run this application, you need the following installed:

Python 3.8 or higher
Required Python libraries:
tensorflow
numpy
librosa
gradio
scikit-learn
You can install the required libraries using the following command:

pip install tensorflow numpy librosa gradio scikit-learn


Installation
Clone the Repository

git clone https://github.com/your-username/speaker-gender-prediction.git
cd speaker-gender-prediction

Add Pre-Trained Models and Label Encoders

Place the following files in the repository root directory:
lstm_speaker_model.h5: Pre-trained speaker recognition model.
lstm_gender_model.h5: Pre-trained gender prediction model.
lstm_speaker_label.pkl: Label encoder for speaker classes.
lstm_gender_label.pkl: Label encoder for gender classes.

Usage
Run the Gradio application:

python app.py


Gradio Interface
The interface allows you to:

Upload an audio file or record audio directly.
Predict the top 3 speakers and their probabilities.
Predict the gender of the speaker.
Handle unknown speakers with a confidence threshold.

Project Structure
.
├── app.py                  # Main application file
├── lstm_speaker_model.h5   # Pre-trained speaker model (to be added)
├── lstm_gender_model.h5    # Pre-trained gender model (to be added)
├── lstm_speaker_label.pkl  # Speaker label encoder (to be added)
├── lstm_gender_label.pkl   # Gender label encoder (to be added)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

Example Output
Top 3 Predicted Speakers:

The top 3 predicted speakers are:
Speaker 1: 85.23%
Speaker 2: 10.12%
Speaker 3: 4.65%

The predicted gender is: 'Gender'

Unknown Speaker:
The top 3 predicted speakers are:
Unknown: 45.23%

The predicted gender is: Unknown


How It Works
Feature Extraction:

Extracts MFCCs, chroma features, and spectral contrast from the input audio file using librosa.
Speaker and Gender Models:

Speaker Model: A pre-trained LSTM model classifies the speaker based on extracted features.
Gender Model: A separate LSTM model determines the gender.
Unknown Detection:

If the highest confidence score for a speaker is below a defined threshold, the speaker is classified as "Unknown."


Roadmap
Add support for real-time audio predictions.
Improve unknown speaker detection using open-set recognition techniques.
Expand the dataset for more robust gender classification.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-branch-name).
Commit your changes (git commit -m "Add new feature").
Push to the branch (git push origin feature-branch-name).
Open a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
TensorFlow: For building the deep learning models.
Librosa: For audio processing and feature extraction.
Gradio: For creating the user interface


