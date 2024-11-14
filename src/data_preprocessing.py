# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data():
    # Replace 'dataset.csv' with your actual dataset file
    data = pd.read_csv('dataset.csv')
    return data

def preprocess_data(data):
    # Handle missing values
    data = data.dropna()

    # Encode gender
    gender_encoder = LabelEncoder()
    data['Gender'] = gender_encoder.fit_transform(data['Gender'])
    joblib.dump(gender_encoder, 'gender_encoder.pkl')  # Save encoder for later use

    # Extract symptoms into individual binary columns
    symptom_list = set()
    for symptoms in data['Symptoms']:
        symptom_list.update(symptoms.split(','))
    symptom_list = list(symptom_list)
    symptom_list.sort()
    joblib.dump(symptom_list, 'symptom_list.pkl')  # Save symptom list for later use

    # One-hot encode symptoms
    for symptom in symptom_list:
        data[symptom] = data['Symptoms'].apply(lambda x: int(symptom in x.split(',')))

    # Drop the original 'Symptoms' column
    data = data.drop(columns=['Symptoms'])

    return data

if __name__ == '__main__':
    data = load_data()
    processed_data = preprocess_data(data)
    processed_data.to_csv('processed_data.csv', index=False)
