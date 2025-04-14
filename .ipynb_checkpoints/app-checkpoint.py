from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
hybrid_model = joblib.load('hybrid_model.pkl')
scaler = joblib.load('scaler.pkl')

# Label mapping
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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_label = None
    activity = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            filename = f"ecg_{datetime.today().strftime('%Y-%m-%d')}.csv"
            df.to_csv(filename, index=False)

            if 'label' in df.columns:
                df = df.drop(columns=['label'])

            df_scaled = scaler.transform(df)
            prediction = hybrid_model.predict(df_scaled)
            prediction_label = prediction[0]
            activity = label_map.get(prediction_label, "Unknown")

    return render_template('index.html', prediction_label=prediction_label,
                           activity=activity, filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
