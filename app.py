import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
st.title("Iris Classifier")

sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)



if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        r.raise_for_status()
        result = r.json()
        st.success(f"Class: {result['class_name']}")
        st.write(f"Probability: {result['probability']}")
    except Exception as e:
        st.error(f"API error: {e}")