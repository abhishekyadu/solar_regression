pip install streamlit pandas scikit-learn matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the trained model
with open('gbr.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for EDA
data = pd.read_csv('solarpowergeneration.csv')

# Streamlit app
st.title("Solar Power Generation Prediction")

st.sidebar.header("Input Features")
# You can customize these input fields based on the actual features used in the model
temp = st.sidebar.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=50)
windspeed = st.sidebar.slider("Wind Speed (m/s)", min_value=0, max_value=20, value=5)
precipitation = st.sidebar.slider("Precipitation (mm)", min_value=0, max_value=50, value=0)

# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[temp, humidity, windspeed, precipitation]])
    prediction = model.predict(input_data)
    st.subheader(f"Predicted Solar Power Generation: {prediction[0]:.2f} MW")

# EDA Section
st.header("Exploratory Data Analysis")

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.subheader("Statistical Summary")
st.write(data.describe())

st.subheader("Correlation Heatmap")
corr = data.corr()
fig, ax = plt.subplots()
cax = ax.matshow(corr, cmap='coolwarm')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
st.pyplot(fig)

st.subheader("Distribution of Solar Power Generation")
fig, ax = plt.subplots()
data['Solar_Power_Generation'].hist(ax=ax, bins=20)
st.pyplot(fig)

# You can add more EDA plots here as needed

