# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd
from nlp_module import extract_symptoms

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('disease_prediction_model.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
symptom_list = joblib.load('symptom_list.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    gender = request.form['gender']
    symptoms_input = request.form['symptoms']

    # Encode gender
    gender_encoded = gender_encoder.transform([gender])[0]

    # Extract symptoms from user input
    extracted_symptoms = extract_symptoms(symptoms_input)

    # Prepare input data
    input_data = pd.DataFrame(columns=['Age', 'Gender'] + symptom_list)
    input_data.at[0, 'Age'] = age
    input_data.at[0, 'Gender'] = gender_encoded

    # Set symptom columns to 0
    for symptom in symptom_list:
        input_data.at[0, symptom] = 0

    # Set extracted symptoms to 1
    for symptom in extracted_symptoms:
        if symptom in symptom_list:
            input_data.at[0, symptom] = 1

    # Handle missing values by filling with 0
    input_data = input_data.fillna(0)

    # Predict disease
    prediction = model.predict(input_data)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
