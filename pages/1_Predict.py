import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Add parent directory to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import encode_inputs, SUBJECT_MAPPING, WEATHER_MAPPING, DAY_MAP

st.set_page_config(page_title="Predict Attendance", page_icon="📈", layout="wide")

st.title("📈 Predict Attendance")
st.markdown("Enter the lecture details below to predict the attendance band (Low, Medium, or High).")

# Load model and encoders
@st.cache_resource
def load_models():
    model = joblib.load("model/xgb-classifier-model.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return model, label_encoder

try:
    model, target_encoder = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    day_of_week = st.selectbox("Day of Week", list(DAY_MAP.keys()))
    subject = st.selectbox("Subject", list(SUBJECT_MAPPING.keys()))
    lecture_number = st.selectbox("Lecture Number", [1, 2, 3, 4, 5])
    start_time = st.selectbox("Start Time", ["8.30 AM", "9.15 AM", "10.15 AM", "11.15 AM", "1.30 PM", "2.30 PM", "3.30 PM"])
    practical_theory = st.radio("Type", ["Theory", "Practical"])
    gap_since_previous = st.selectbox("Gap Since Previous Lecture", ["Same Day", "1 Day", "2 Days", "3 Days", "4 Days", "5 Days", "6 Days", "7 Days"])
    week_number = st.number_input("Week Number of Semester", min_value=1, max_value=20, value=1)

with col2:
    internal_test_week = st.radio("Internal Test This Week?", ["No", "Yes"])
    assignment_due = st.radio("Assignment Due?", ["No", "Yes"])
    holiday_before_after = st.selectbox("Holiday Before/After?", ["No", "Before", "After"])
    weather = st.selectbox("Weather", list(WEATHER_MAPPING.keys()))
    special_event = st.radio("Special Event on Campus?", ["No", "Yes"])
    previous_attendance = st.slider("Previous Lecture Attendance", min_value=0, max_value=200, value=50)

# Submit button
if st.button("Predict Attendance 🚀", use_container_width=True):
    # Collect inputs
    inputs = {
        'day_of_week': day_of_week,
        'subject': subject,
        'lecture_number': lecture_number,
        'start_time': start_time,
        'practical_theory': practical_theory,
        'gap_since_previous': gap_since_previous,
        'week_number': week_number,
        'internal_test_week': internal_test_week,
        'assignment_due': assignment_due,
        'holiday_before_after': holiday_before_after,
        'weather': weather,
        'special_event': special_event,
        'previous_lecture_attendance': previous_attendance
    }
    
    with st.spinner("Analyzing parameters..."):
        try:
            # 1. Encode Inputs (passes dataset path to calculate rolling avg)
            dataset_path = "data/attendance_dataset-V2.csv"
            df_encoded = encode_inputs(inputs, dataset_path)
            
            # 2. Predict
            # XGBClassifier predict_proba returns probabilities for each class
            prediction_idx = model.predict(df_encoded)[0]
            prediction_probs = model.predict_proba(df_encoded)[0]
            
            # Inverse transform index to Label
            prediction_label = target_encoder.inverse_transform([prediction_idx])[0]
            
            # 3. Display Result
            st.markdown("### Prediction Result")
            
            if prediction_label == "High":
                st.success("🌟 The model predicts **High** attendance for this lecture.")
            elif prediction_label == "Medium":
                st.warning("⚠️ The model predicts **Medium** attendance for this lecture.")
            else:
                st.error("📉 The model predicts **Low** attendance for this lecture.")
            
            # Show probabilities
            st.markdown("#### Prediction Probabilities")
            classes = target_encoder.classes_ # e.g., ['High', 'Low', 'Medium']
            
            for cls, prob in zip(classes, prediction_probs):
                st.write(f"**{cls}** ({prob*100:.1f}%)")
                st.progress(float(prob))
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
