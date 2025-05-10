# Human Activity Recognition using Machine Learning & Deep Learning

This project aims to classify human activities based on sensor data collected from wearable devices. The solution integrates feature selection using the Wolf Optimization Algorithm, a Random Forest classifier, and a deep neural network for robust performance. A user-friendly Gradio interface is included for real-time predictions.

## Project Structure

human-activity-recognition/
├── data/
│ ├── train.csv
│ └── test.csv
├── models/
│ ├── classification_model.h5
│ ├── label_encoder.pkl
│ └── scaler.pkl
├── src/
│ ├── feature_selection.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── app_gradio.py
├── requirements.txt
└── README.md


## Features

- **Feature Selection:** Wolf Optimization Algorithm (WOA)
- **Machine Learning:** Random Forest for preliminary accuracy evaluation
- **Deep Learning:** Multi-layer neural network using TensorFlow/Keras
- **Evaluation:** Confusion matrix and classification report
- **Deployment:** Gradio interface for activity prediction


## Dataset
The dataset used in this project is based on the UCI Human Activity Recognition dataset, which includes sensor signals collected from smartphones worn by volunteers performing different activities.

## Requirements

Python 3.8+

TensorFlow

scikit-learn

pandas

numpy

joblib

gradio

seaborn

matplotlib

## Model Performance
The model achieves high accuracy by leveraging optimized feature selection and deep learning techniques. Performance is validated using cross-validation and confusion matrix analysis.
