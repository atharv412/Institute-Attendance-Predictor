import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Time Slots Analysis", page_icon="⏰", layout="wide")

st.title("⏰ Time Slots Analysis")
st.markdown("Explore historical attendance patterns purely based on **Day of Week**, **Lecture Number**, and **Start Time** to identify consistently underperforming slots. *(This page is purely analytical and does not use the predictive model).*")

@st.cache_data
def load_data():
    # Read the dataset directly
    return pd.read_csv("data/attendance_dataset-V3_500_rows.csv")

try:
    df = load_data()
    
    # Group by Day, Lecture Number, and Start Time
    grouped = df.groupby(['Day_of_Week', 'Lecture_Number', 'Start_Time'])['Attendance_Percentage'].mean().reset_index()
    
    # Order the days of the week logically
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    # Ensure we only include days that are actually in the dataset
    existing_days = [d for d in day_order if d in grouped['Day_of_Week'].unique()]
    
    grouped['Day_of_Week'] = pd.Categorical(grouped['Day_of_Week'], categories=existing_days, ordered=True)
    grouped = grouped.sort_values(['Day_of_Week', 'Lecture_Number'])
    
    # Prepare data for the heatmap
    pivot_df = grouped.pivot(index='Day_of_Week', columns='Lecture_Number', values='Attendance_Percentage')
    
    # Format column names for the x-axis
    pivot_df.columns = [f"Lecture {c}" for c in pivot_df.columns]
    
    st.markdown("### 📊 Heatmap of Average Attendance")
    
    # Create the heatmap using Plotly
    fig = px.imshow(
        pivot_df,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        labels=dict(x="Lecture Slot", y="Day of Week", color="Avg Attendance %"),
        height=500
    )
    
    # Add annotations/borders for any cell below 50%
    for day in pivot_df.index:
        for lecture in pivot_df.columns:
            val = pivot_df.loc[day, lecture]
            if pd.notnull(val) and val < 50.0:
                fig.add_annotation(
                    x=lecture,
                    y=day,
                    text="⚠️ < 50%",
                    showarrow=False,
                    font=dict(color="red", size=10, weight="bold"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="red",
                    borderwidth=1,
                    borderpad=3,
                    yshift=-25  # Shift down so it doesn't cover the main number
                )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.markdown("### 📉 Worst Performing Time Slots")
    st.markdown("The following table ranks the time slots from worst average attendance to best.")
    
    # Sort grouped data by Attendance_Percentage ascending for the dataframe
    worst_slots = grouped.sort_values('Attendance_Percentage', ascending=True)
    
    # Rename columns for a cleaner UI
    worst_slots = worst_slots.rename(columns={
        'Day_of_Week': 'Day of Week',
        'Lecture_Number': 'Lecture Number',
        'Start_Time': 'Start Time',
        'Attendance_Percentage': 'Avg Attendance (%)'
    })
    
    # Format the percentage to 2 decimal places
    worst_slots['Avg Attendance (%)'] = worst_slots['Avg Attendance (%)'].round(2)
    
    # Display the dataframe with the worst slots at the top
    st.dataframe(worst_slots, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"An error occurred while generating the analytics: {e}")
