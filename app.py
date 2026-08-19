import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Attendance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Welcome to the Attendance Prediction Dashboard")

st.markdown("""
Welcome! This application uses an XGBoost machine learning model to predict classroom attendance 
based on various scheduling and environmental factors. Use the sidebar to navigate to the **Predict** 
page, where you can enter the details of a lecture to see if the expected attendance will be **Low**, **Medium**, or **High**.
""")

st.divider()
st.subheader("📊 Dataset Overview")

@st.cache_data
def load_data():
    return pd.read_csv("data/attendance_dataset-V3_500_rows.csv")

try:
    df = load_data()
    
    # Calculate metrics
    total_lectures = len(df)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    min_date = df['Date'].min().strftime('%d %b %Y')
    max_date = df['Date'].max().strftime('%d %b %Y')
    
    avg_attendance = df['Attendance_Percentage'].mean()
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Lectures Recorded", f"{total_lectures:,}")
    col2.metric("Date Range", f"{min_date} to {max_date}")
    col3.metric("Overall Average Attendance", f"{avg_attendance:.2f}%")
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Show some random insights / cards
    st.subheader("💡 Key Insights")
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.info("🕒 **Time Matters:** The model identifies a notable drop in attendance for afternoon sessions starting at 1:30 PM.")
        st.success("🌦️ **Weather Impact:** Weather conditions like Rain vs. Sun can subtly shift attendance likelihood across subjects.")
    
    with insight_col2:
        st.warning("📅 **Holidays & Tests:** Weeks with internal tests or adjacent to holidays tend to show highly volatile attendance.")
        st.info("🔄 **Momentum:** The 'Rolling Average' of the last 3 lectures is one of the strongest predictors of future attendance.")

except Exception as e:
    st.error(f"Error loading dataset: {e}")
