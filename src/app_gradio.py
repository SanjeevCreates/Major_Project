import gradio as gr
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('models/classification_model.h5')
label_encoder = joblib.load('models/label_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')

def predict_activity(*args):
    input_data = np.array(args).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction_probs = model.predict(input_scaled)
    prediction = np.argmax(prediction_probs, axis=1)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

inputs = [
    gr.Number(label='tBodyAcc-mean()-X'),
    gr.Number(label='tBodyAcc-mean()-Y'),
    gr.Number(label='tBodyAcc-mean()-Z'),
    gr.Number(label='tBodyAcc-std()-X'),
    gr.Number(label='tBodyAcc-std()-Z'),
    gr.Number(label='tBodyAcc-mad()-Y'),
    gr.Number(label='tBodyAcc-max()-X'),
    gr.Number(label='tBodyAcc-max()-Y'),
    gr.Number(label='tBodyAcc-energy()-X'),
    gr.Number(label='tBodyAcc-energy()-Z'),
    gr.Number(label='tBodyAcc-iqr()-Z'),
    gr.Number(label='tBodyAcc-entropy()-X'),
    gr.Number(label='tBodyAcc-entropy()-Y'),
    gr.Number(label='tBodyAcc-arCoeff()-X,1'),
    gr.Number(label='tBodyAcc-arCoeff()-X,3'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,1'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,2'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,3'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,4'),
    gr.Number(label='tBodyAcc-arCoeff()-Z,1')
]

interface = gr.Interface(
    fn=predict_activity,
    inputs=inputs,
    outputs="text",
    title="Human Activity Recognition",
    description="Enter the feature values and click 'Submit' to predict the activity class.",
    live=False,
)

interface.launch(share=True)


