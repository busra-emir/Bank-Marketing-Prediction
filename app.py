import streamlit as st
import pandas as pd
import joblib
import random

# Import required libraries at the top
try:
    import pandas as pd
    import joblib
    import random
except ImportError as e:
    st.error(f"Required library missing: {e}")
    st.stop()

st.set_page_config(
    page_title="BSA Model Interactive Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the sidebar buttons
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-bottom: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        color: #000000;
        text-align: left;
        padding: 15px;
    }
    .stButton > button:hover {
        background-color: #e0e2e6;
    }
    .selected-button > button {
        background-color: #00acb5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load common resources
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bank-additional.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('BSA_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar navigation
st.sidebar.title("Navigation")

# Initialize session state for current page if not exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Navigation buttons
col1 = st.sidebar.columns(1)[0]
with col1:
    if st.button("🏠 Home", key="home", 
                help="Return to home page",
                use_container_width=True):
        st.session_state.current_page = "Home"
        
    if st.button("📊 Data Preview", key="data", 
                help="View dataset preview",
                use_container_width=True):
        st.session_state.current_page = "Data Preview"
        
    if st.button("📋 Model Details", key="model", 
                help="View model information",
                use_container_width=True):
        st.session_state.current_page = "Model Details"
        
    if st.button("⚙️ Make Prediction", key="predict", 
                help="Make new predictions",
                use_container_width=True):
        st.session_state.current_page = "Make Prediction"
        
    if st.button("🔍 Ask Model", key="ask", 
                help="Test model with random samples",
                use_container_width=True):
        st.session_state.current_page = "Ask Model"

# Load data and model once
data = load_data()
model = load_model()

# Content based on selected page
if st.session_state.current_page == "Home":
    st.title("BSA Model Interactive Demo")
    st.write("This application uses a **Random Forest** model to make predictions based on the **bank-additional.csv** dataset.")
    
    st.markdown("""
    ## Welcome to the BSA Model Demo!

    This application demonstrates a machine learning model for bank customer analysis. You can:

    1. 📊 View and explore the dataset in the **Data Preview** section
    2. 📋 Learn about the model in the **Model Details** section
    3. ⚙️ Make predictions with your own input in the **Make Prediction** section
    4. 🔍 Test the model with random samples in the **Ask Model** section

    Please select a section from the sidebar to get started!
    """)

elif st.session_state.current_page == "Data Preview":
    st.title("Data Preview")
    if data is not None:
        st.write("Here are the first 5 rows of the dataset:")
        st.dataframe(data.head())
        st.subheader("Dataset Summary")
        st.write(f"Total Rows: {data.shape[0]}, Total Columns: {data.shape[1]}")
        st.write("Column Information:")
        st.write(data.dtypes)

elif st.session_state.current_page == "Model Details":
    st.title("Model Details")
    st.subheader("Model Information")
    st.write("The **Random Forest** model was used for this application. Below are the details of the model:")
    st.text("Model File: BSA_model.pkl")
    st.text("Model Objective: Predict customer behavior based on banking data.")

    # Display feature importance visualization
    st.write("### Feature Importance")
    importance_data = {
        'Feature': ['euribor3m', 'age', 'job', 'nr.employed', 'campaign', 'education', 
                   'day_of_week', 'emp.var.rate', 'marital', 'cons.conf.idx'],
        'Importance': [0.16, 0.15, 0.08, 0.08, 0.08, 0.08, 0.07, 0.04, 0.04, 0.03]
    }
    importance_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=True)
    st.bar_chart(importance_df.set_index('Feature'))

elif st.session_state.current_page == "Make Prediction":
    st.title("Make a Prediction")
    
    if data is not None and model is not None:
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Primary Features")
            # Most important features according to the graph
            euribor3m = st.number_input("Euribor 3 Month Rate", 
                                      min_value=-5.0, max_value=10.0, value=1.0, step=0.1,
                                      help="Euro Interbank Offered Rate - 3 month rate")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            job = st.selectbox("Job", options=["admin.", "blue-collar", "entrepreneur", "housemaid", 
                                             "management", "retired", "self-employed", "services", 
                                             "student", "technician", "unemployed", "unknown"])
            nr_employed = st.number_input("Number of Employees", 
                                        min_value=1000, max_value=10000, value=5000,
                                        help="Number of employees - quarterly indicator")
            
        with col2:
            st.subheader("Secondary Features")
            campaign = st.number_input("Number of Contacts", min_value=1, max_value=50, value=1,
                                     help="Number of contacts performed during this campaign for this client")
            education = st.selectbox("Education", options=["basic.4y", "basic.6y", "basic.9y", 
                                                         "high.school", "illiterate", 
                                                         "professional.course", "university.degree", 
                                                         "unknown"])
            day_of_week = st.selectbox("Day of Week", options=["mon", "tue", "wed", "thu", "fri"])
            emp_var_rate = st.number_input("Employment Variation Rate", 
                                         min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
                                         help="Employment variation rate - quarterly indicator")
            
        # Less important features in an expander
        with st.expander("Additional Features"):
            marital = st.selectbox("Marital Status", options=["married", "divorced", "single"])
            cons_conf_idx = st.number_input("Consumer Confidence Index", 
                                          min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
            
        # Create input dictionary with all required features
        input_data = {
            'euribor3m': euribor3m,
            'age': age,
            'job': job,
            'nr.employed': nr_employed,
            'campaign': campaign,
            'education': education,
            'day_of_week': day_of_week,
            'emp.var.rate': emp_var_rate,
            'marital': marital,
            'cons.conf.idx': cons_conf_idx,
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add debug information
        with st.expander("Debug Information"):
            st.write("Input Features:")
            st.write(input_df.columns.tolist())
            if model is not None:
                st.write("Expected Features:")
                if hasattr(model, 'feature_names_in_'):
                    st.write(model.feature_names_in_.tolist())
        
        # Prediction Button
        if st.button("Make Prediction"):
            try:
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
                
                # Display prediction with probability
                if prediction[0] == 'yes':
                    st.success(f"Prediction: Customer is likely to subscribe! ✅ \
                             (Confidence: {prediction_proba[0][1]:.2%})")
                else:
                    st.info(f"Prediction: Customer is unlikely to subscribe ❌ \
                           (Confidence: {prediction_proba[0][0]:.2%})")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.error("Please ensure all required features are provided and in the correct format.")

elif st.session_state.current_page == "Ask Model":
    st.title("Ask the Model")
    
    if data is not None and model is not None:
        st.write("This section allows you to randomly select a row from the dataset and let the model predict the outcome.")
        
        if st.button("Select Random Row"):
            random_index = random.randint(0, len(data) - 1)
            selected_row = data.iloc[random_index]
            
            st.write("### Selected Row:")
            st.write(selected_row)
            
            # Prepare input for prediction
            input_data = selected_row.drop('y')
            input_df = input_data.to_frame().T
            
            try:
                prediction = model.predict(input_df)
                st.success(f"Model Prediction: {prediction[0]}")
                st.info(f"Actual Value: {selected_row['y']}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
