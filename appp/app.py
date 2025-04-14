import gradio as gr
import pandas as pd
import joblib
from datetime import datetime

# Load the model and scaler
hybrid_model = joblib.load("hybrid_model.pkl")
scaler = joblib.load("scaler.pkl")

# Activity label map
label_map = {
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouching)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back"
}

def predict_activity(file):
    df = pd.read_csv(file.name)
    filename = f"ecg_{datetime.today().strftime('%Y-%m-%d')}.csv"
    df.to_csv(filename, index=False)

    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    df_scaled = scaler.transform(df)
    prediction = hybrid_model.predict(df_scaled)
    activity = label_map.get(prediction[0], "Unknown")

    return f"Predicted activity: {activity}", filename

# Gradio UI
demo = gr.Interface(
    fn=predict_activity,
    inputs=gr.File(label="Upload ECG CSV"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.File(label="Download Saved CSV")
    ],
    title="ECG Activity Recognition",
    description="Upload your ECG CSV file to predict the activity."
)

if __name__ == "__main__":
    demo.launch()
