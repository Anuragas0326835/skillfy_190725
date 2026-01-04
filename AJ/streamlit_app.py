import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
MODEL_PATH = "random_forest_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please train and save the model first.")
    st.stop()
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Define feature names (from the winequality-red.csv dataset)
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

st.title("Wine Quality Prediction")

st.write("Enter the wine characteristics below to predict its quality:")

# Create input fields for each feature
input_data = {}
for feature in FEATURES:
    input_data[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.01)

if st.button("Predict Quality"):
    # Prepare input for prediction
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Wine Quality: {prediction}")