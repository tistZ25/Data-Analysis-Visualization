import streamlit as st
import numpy as np
import joblib

scaler = joblib.load('Scaler.pkl')
model = joblib.load('model.pkl')

st.title('Kolkata House Price Prediction')



st.divider()

bhk = st.number_input('Enter Number of Bedrroms (BHK)', value = 2, step = 1)
bath = st.number_input('Enter Number of Bathrooms', value = 1, step = 1)
size = st.number_input('Enter Size in SqftArea:', value = 1000, step = 50)
price_sqft = st.number_input('Enter Price per Square Foot (Optional):', min_value=0, step=1)

X = [bhk, bath, size, price_sqft]


st.divider()

predict_button = st.button('Predict!')



st.divider()

if predict_button:
    st.balloons()
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    st.write(f'The Prediction is {prediction:.2f}')

else:
    'Please use the button for prediction!'
