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

st.markdown("### 🤖 Select a Model")
model_choice = st.selectbox(
    "Choose the machine learning model to use for prediction:",
    options=["XGBoost", "Random Forest", "Logistic Regression"],
    help="Select the model you'd like to use. XGBoost is highly optimized and accurate. Random Forest provides robust predictions. Logistic Regression acts as a solid, interpretable baseline."
)

MODEL_PATHS = {
    "XGBoost": "model/XGBoost/xgb-classifier-model.pkl",
    "Random Forest": "model/Random_Forest/random_forest-classifier-model.pkl",
    "Logistic Regression": "model/Logistic_Regression/logistic-regressor-classifier-model.pkl"
}

MODEL_METRICS = {
    "XGBoost": {"Accuracy": "61.0%", "Precision": "60.0%"},
    "Random Forest": {"Accuracy": "48.0%", "Precision": "48.0%"},
    "Logistic Regression": {"Accuracy": "56.0%", "Precision": "56.0%"}
}

REGRESSOR_METRICS = {
    "MAPE": "32.28%",
    "R²": "0.335"
}

# Load model and encoders based on selection
@st.cache_resource
def load_models(selected_model):
    model = joblib.load(MODEL_PATHS[selected_model])
    
    label_encoder = None
    # Only XGBoost requires the external label encoder; Scikit-Learn models return string labels natively.
    if selected_model == "XGBoost":
        label_encoder = joblib.load("model/XGBoost/label_encoder.pkl")
        
    return model, label_encoder

@st.cache_resource
def load_regressor():
    return joblib.load("model/GradientBoosting/gradient-Boosting-regressor-model.pkl")

try:
    model, target_encoder = load_models(model_choice)
    regressor_model = load_regressor()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()
    
st.divider()
st.markdown("### 📝 Enter Lecture Details")

# Layout
col1, col2 = st.columns(2)

with col1:
    day_of_week = st.selectbox("Day of Week", list(DAY_MAP.keys()))
    
    BASE_SUBJECTS = [
        "Data Science & Machine Learning",
        "Industry Readiness Program",
        "Innovation and Entrepreneurship Development",
        "Mini Project",
        "Mobile Application Development",
        "Principles of Cloud Management and Security",
        "Software Testing and Quality Assurance"
    ]
    subject = st.selectbox("Subject", BASE_SUBJECTS)
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
    # Map base subject + type to the exact dataset subject string
    actual_subject = subject
    if practical_theory == "Practical":
        if subject == "Data Science & Machine Learning":
            actual_subject = "DS & ML Practical"
        elif subject == "Mobile Application Development":
            actual_subject = "MAD Practical"
        elif subject == "Software Testing and Quality Assurance":
            actual_subject = "STQA Practical"

    # Collect inputs
    inputs = {
        'day_of_week': day_of_week,
        'subject': actual_subject,
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
            dataset_path = "data/cleaned_attendance_dataset.csv"
            df_encoded = encode_inputs(inputs, dataset_path)
            
            # 2. Predict
            prediction_raw = model.predict(df_encoded)[0]
            prediction_probs = model.predict_proba(df_encoded)[0]
            
            # XGBoost returns an integer index, so we inverse_transform it.
            # Scikit-Learn models (RF/LogReg) return the string directly.
            if target_encoder is not None:
                prediction_label = target_encoder.inverse_transform([int(prediction_raw)])[0]
            else:
                prediction_label = str(prediction_raw)
            
            # 3. Predict with Regressor
            predicted_value = regressor_model.predict(df_encoded)[0]
            
            # 4. Display Results and Metrics
            st.markdown("### Prediction Result & Model Metrics")
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.markdown(f"**{model_choice} Metrics:**")
                st.write(f"- Accuracy: `{MODEL_METRICS[model_choice]['Accuracy']}`")
                st.write(f"- Precision: `{MODEL_METRICS[model_choice]['Precision']}`")
            with metrics_col2:
                st.markdown("**GradientBoosting Regressor Metrics:**")
                st.write(f"- MAPE: `{REGRESSOR_METRICS['MAPE']}`")
                st.write(f"- R² Score: `{REGRESSOR_METRICS['R²']}`")
            
            st.divider()
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("#### 📊 Categorical Band")
                if prediction_label == "High":
                    st.success("🌟 The classifier predicts **High** attendance.")
                elif prediction_label == "Medium":
                    st.warning("⚠️ The classifier predicts **Medium** attendance.")
                else:
                    st.error("📉 The classifier predicts **Low** attendance.")
                
                st.markdown("**Probabilities:**")
                classes = target_encoder.classes_ if target_encoder is not None else model.classes_
                for cls, prob in zip(classes, prediction_probs):
                    st.write(f"**{cls}** ({prob*100:.1f}%)")
                    st.progress(float(prob))
            
            with res_col2:
                st.markdown("#### 🎯 Numerical Estimate")
                st.info(
                    f"**Estimated Attendance:** ~{int(predicted_value)} students\n\n"
                    f"**Confidence Range:** {max(0, int(predicted_value - 11))} to {int(predicted_value + 11)} students"
                )
                st.markdown(
                    "*(The GradientBoosting Regressor provides a specific numerical estimate with an average error of ±11 students).* "
                )
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
