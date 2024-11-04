import streamlit as st
import pandas as pd
import joblib  # or you can use pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load('model_chrun.pkl')

# Set up the app title and description
st.title("Customer Churn Prediction")
st.write("Upload customer data to predict churn likelihood.")

# File uploader to get CSV input from the user
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Summary:")
    st.write(df.describe())

    # Visualize the Churn distribution if 'Churn' column is available
    if 'Churn' in df.columns:
        st.write("Churn Distribution:")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)

    # Preprocess the data (handle missing values, scaling, encoding, etc.)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Scaling numerical features (example)
    scaler = StandardScaler()
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Make predictions using the loaded model
    if st.button("Predict Churn"):
        predictions = model.predict(df[numerical_features])
        df['Predicted Churn'] = predictions
        st.write("Predictions:")
        st.write(df[['customerID', 'Predicted Churn']])

        # Show counts of predicted churn
        st.write("Predicted Churn Distribution:")
        fig, ax = plt.subplots()
        sns.countplot(x='Predicted Churn', data=df, ax=ax)
        st.pyplot(fig)

