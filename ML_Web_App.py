import streamlit as st
import numpy as np
import joblib

# Load the scaler and model
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

# Set up the Streamlit app
st.title("Real Estate Price Prediction App")
st.divider()

# Input fields for user
bed = st.number_input("Enter the number of bedrooms", value=2, step=1)
bath = st.number_input("Enter the number of bathrooms", value=1, step=1)
size = st.number_input("Enter the size (in square feet)", value=1000, step=50)

# Prepare the input array
x = [bed, bath, size]
x1 = np.array(x).reshape(1, -1)

st.divider()

# Prediction button
predict_button = st.button("Predict!")
st.divider()

# Handle prediction
if predict_button:
    st.balloons()
    x_array = scaler.transform(x1)
    prediction = model.predict(x_array)[0]
    st.write(f"The predicted price is ${prediction:.2f}")
else:
    st.write("Please use the button for prediction")
