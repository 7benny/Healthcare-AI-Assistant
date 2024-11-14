# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_data():
    data = pd.read_csv('processed_data.csv')
    return data

def train_model():
    data = load_data()

    # Features and target variable
    X = data.drop(columns=['Disease'])
    y = data['Disease']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, 'disease_prediction_model.pkl')

if __name__ == '__main__':
    train_model()
