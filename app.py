import os
import streamlit as st
import pandas as pd
import joblib
import random

# ----------------------------
# File Paths
# ----------------------------
model_path = 'BSA_model.pkl'
data_path = 'bank-additional.csv'

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="BSA Model Interactive Demo", layout="wide")

# Title
st.title("BSA Model Interactive Demo")
st.write("This application uses a **Random Forest** model to make predictions based on the **bank-additional.csv** dataset.")

# ----------------------------
# Load Data and Model
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

@st.cache_resource
def load_model():
    return joblib.load(model_path)

# Load the data and model
try:
    data = load_data()
    model = load_model()
    st.success("Model and data loaded successfully!")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ----------------------------
# Tabs: Data Preview | Model Details | Make a Prediction | Ask the Model
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "📋 Model Details", "⚙️ Make a Prediction", "🔍 Ask the Model"])

# ----------------------------
# Tab 1: Data Preview
# ----------------------------
with tab1:
    st.subheader("Data Preview")
    st.write("Here are the first 5 rows of the **bank-additional.csv** dataset:")
    st.dataframe(data.head())

    # Data Summary
    st.subheader("Dataset Summary")
    st.write(f"Total Rows: {data.shape[0]}, Total Columns: {data.shape[1]}")
    st.write("Column Information:")
    st.write(data.dtypes)

# ----------------------------
# Tab 2: Model Details
# ----------------------------
with tab2:
    st.subheader("Model Details")
    st.write("The **Random Forest** model was used for this application. Below are the details of the model:")
    st.text("Model File: BSA_model.pkl")
    st.text("Model Size: 20.2 MB")
    st.text("Model Objective: Predict customer behavior based on banking data.")
    st.write("You can include model performance metrics here (e.g., accuracy, F1-score).")

# ----------------------------
# Tab 3: Make a Prediction
# ----------------------------
with tab3:
    st.subheader("Make a Prediction")

    # Collect User Input
    st.write("Please fill in the following fields to make a prediction:")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", options=["admin.", "technician", "services", "management", "unemployed"])
    balance = st.number_input("Balance", min_value=-5000, max_value=50000, value=1000)
    duration = st.slider("Call Duration (seconds)", min_value=0, max_value=5000, value=300)
    
    # Example Input Data
    input_data = pd.DataFrame({
        "age": [age],
        "job": [job],
        "balance": [balance],
        "duration": [duration]
    })

    # Prediction Button
    if st.button("Make Prediction"):
        try:
            # Get Prediction from the Model
            prediction = model.predict(input_data)
            st.success(f"Model Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# ----------------------------
# Tab 4: Ask the Model
# ----------------------------
with tab4:
    st.subheader("Ask the Model")

    # Instruction
    st.write("This section allows you to randomly select a row from the dataset and let the model predict the outcome.")

    # Random Row Selection
    if st.button("Select Random Row"):
        random_index = random.randint(0, len(data) - 1)
        selected_row = data.iloc[random_index]
        st.write("### Selected Row:")
        st.write(selected_row)

        # Extract input features for the model
        feature_columns = ["age", "job", "balance", "duration"]  # Adjust these columns to match your model
        input_data = selected_row[feature_columns].to_frame().T

        # Show input data
        st.write("### Input Features for the Model:")
        st.dataframe(input_data)

        # Model Prediction
        try:
            prediction = model.predict(input_data)
            st.success(f"Model Prediction: {prediction[0]}")
            # Compare with the actual value if target column exists
            if "target" in data.columns:  # Replace 'target' with the actual target column name
                actual_value = selected_row["target"]
                st.info(f"Actual Value: {actual_value}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
