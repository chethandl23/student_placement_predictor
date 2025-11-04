import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os # Added for path handling

# Define the relative path to the models folder
MODEL_DIR = r'C:\code crafters\backend\models'

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model and the preprocessor."""
    
    # NEW: Use os.path.join to construct platform-independent paths
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')
    
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except FileNotFoundError:
        # NEW: Improved error message with checked paths
        st.error(f"Model or Preprocessor file not found.")
        st.info(f"The application checked for files in the **'{MODEL_DIR}'** folder: `{model_path}` and `{preprocessor_path}`.")
        st.info("Please ensure you have successfully run `data_processor.py` and `model_trainer.py` and that they created the files in the correct location.")
        return None, None

model, preprocessor = load_artifacts()

# --- 2. Define UI Input Options (Code unchanged) ---
GENDER_OPTIONS = ['Female', 'Male']
BOARD_10TH_OPTIONS = ['CBSE', 'ICSE', 'State Board', 'WBBSE']
BOARD_12TH_OPTIONS = ['CBSE', 'WBCHSE', 'Other state Board', 'Diploma', 'ISC', 'CISCE', 'MSBTE', 'BSEB', 'WBBSE', 'ISE', 'Diploma board - MSBTE']
STREAM_OPTIONS = [
    'Artificial Intelligence and Data Science',
    'Artificial Intelligence and Machine Learning',
    'Information Technology',
    'Computer Science',
    'Cyber Security Engineer'
]
YES_NO_OPTIONS = ['Yes', 'No']
COMMUNICATION_LEVEL_MAP = {
    1: 'Poor', 2: 'Average', 3: 'Good', 4: 'Excellent'
}
TECHNICAL_COURSE_MAP = {
    1: 'Basic', 2: 'Intermediate', 3: 'Advanced', 4: 'Expert'
}


def make_prediction(input_data):
    """Preprocesses the input data and returns the placement probability."""
    # Convert input dict to a DataFrame
    input_df = pd.DataFrame([input_data])
    processed_input = preprocessor.transform(input_df)
    placement_probability = model.predict_proba(processed_input)[:, 1][0]
    return placement_probability

# --- 3. Streamlit Application (Code unchanged) ---
st.set_page_config(page_title="Placement Probability Predictor", layout="wide")

st.title("Student Placement Predictor")
st.markdown("Enter the student's details to predict their probability of getting placed.")

if model and preprocessor:
    # --- Input Fields Layout ---
    with st.form("prediction_form"):
        st.header("Academic Details")
        col1, col2, col3 = st.columns(3)

        # Numerical Inputs
        cgpa = col1.slider("CGPA (0 to 10)", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
        marks_10th = col2.number_input("10th Marks (%)", min_value=50.0, max_value=100.0, value=85.0, step=0.1)
        marks_12th = col3.number_input("12th Marks (%)", min_value=40.0, max_value=100.0, value=75.0, step=0.1)

        st.divider()
        st.header("Core Information")
        col4, col5, col6 = st.columns(3)

        # Categorical Inputs
        gender = col4.selectbox("Gender", options=GENDER_OPTIONS)
        stream = col5.selectbox("Stream", options=STREAM_OPTIONS)
        backlog = col6.number_input("Backlogs in 5th Sem", min_value=0, max_value=5, value=0, step=1)
        
        st.divider()
        st.header("Extra-Curricular & Skills")
        col7, col8, col9 = st.columns(3)
        
        internships = col7.selectbox("Internships Done?", options=YES_NO_OPTIONS)
        training = col7.selectbox("Training/Certification Done?", options=YES_NO_OPTIONS)
        innovative_project = col7.selectbox("Innovative Project Done?", options=YES_NO_OPTIONS)

        # Mapped Inputs
        comm_level_key = col8.selectbox("Communication Level", options=list(COMMUNICATION_LEVEL_MAP.keys()), format_func=lambda x: COMMUNICATION_LEVEL_MAP[x])
        tech_course_key = col9.selectbox("Technical Course Level", options=list(TECHNICAL_COURSE_MAP.keys()), format_func=lambda x: TECHNICAL_COURSE_MAP[x])
        
        st.divider()
        st.header("Board Information")
        col10, col11 = st.columns(2)
        
        board_10th = col10.selectbox("10th Board", options=BOARD_10TH_OPTIONS)
        board_12th = col11.selectbox("12th Board", options=BOARD_12TH_OPTIONS)
        
        # Submission button
        submitted = st.form_submit_button("Predict Placement Probability")

    if submitted:
        input_data = {
            'Gender': gender,
            '10th board': board_10th,
            '10th marks': marks_10th,
            '12th board': board_12th,
            '12th marks': marks_12th,
            'Stream': stream,
            'Cgpa': cgpa,
            'Internships(Y/N)': internships,
            'Training(Y/N)': training,
            'Backlog in 5th sem': float(backlog),
            'Innovative Project(Y/N)': innovative_project,
            'Communication level': float(comm_level_key),
            'Technical Course': float(tech_course_key)
        }

        prob = make_prediction(input_data)
        prob_percent = prob * 100

        st.subheader("Prediction Result")
        st.metric(label="Placement Probability", value=f"{prob_percent:.2f}%")

        if prob >= 0.70:
            st.success("High probability of placement! Keep up the good work.")
        elif prob >= 0.50:
            st.warning("Moderate probability. Focus on enhancing skills and communication.")
        else:
            st.error("Lower probability. Prioritize clearing backlogs, doing internships, and improving technical courses.")
            
else:
    st.info("The prediction engine is not fully loaded. Please ensure `data_processor.py` and `model_trainer.py` were executed successfully.")