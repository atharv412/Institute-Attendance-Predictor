import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Subject Analysis", page_icon="📚", layout="wide")

st.title("📚 Subject Analysis")
st.markdown("Analyze historical attendance on a per-subject basis to identify courses with the lowest engagement. ")

@st.cache_data
def load_data():
    df = pd.read_csv("data/attendance_dataset-V3_500_rows.csv")
    # Dynamically compute the bands using the same 3-quantile logic from our data pipeline
    df['Attendance_Band'] = pd.qcut(df['Attendance_Percentage'], q=3, labels=['Low', 'Medium', 'High'])
    return df

try:
    df = load_data()
    
    # 1. Subject Filter
    all_subjects = sorted(df['Subject'].unique().tolist())
    
    st.markdown("### 🔍 Filter Subjects")
    selected_subjects = st.multiselect(
        "Select specific subjects to focus on:", 
        options=all_subjects, 
        default=all_subjects
    )
    
    if not selected_subjects:
        st.warning("Please select at least one subject to view the charts.")
        st.stop()
        
    filtered_df = df[df['Subject'].isin(selected_subjects)]
    
    # 2. Data Aggregation for Bar Chart
    # Compute mean attendance and the percentage of lectures that fell into the "Low" band
    subject_stats = filtered_df.groupby('Subject').agg(
        Mean_Attendance=('Attendance_Percentage', 'mean'),
        Low_Band_Count=('Attendance_Band', lambda x: (x == 'Low').sum()),
        Total_Lectures=('Attendance_Percentage', 'count')
    ).reset_index()
    
    subject_stats['Low_Band_Percentage'] = (subject_stats['Low_Band_Count'] / subject_stats['Total_Lectures']) * 100
    
    # Sort by Mean Attendance (Descending). Plotly draws horizontal bars from bottom to top, 
    # so sorting descending ensures the worst subjects (lowest mean) appear at the TOP of the chart.
    subject_stats = subject_stats.sort_values('Mean_Attendance', ascending=False)
    ordered_subjects = subject_stats['Subject'].tolist()
    
    # 3. Horizontal Bar Chart (Mean Attendance vs Low Band %)
    st.divider()
    st.markdown("### 📊 Average Attendance & Low Band Frequency")
    
    # Melt the dataframe to plot two metrics side-by-side using Plotly Express grouped bar chart
    melted_stats = subject_stats.melt(
        id_vars='Subject', 
        value_vars=['Mean_Attendance', 'Low_Band_Percentage'], 
        var_name='Metric', 
        value_name='Percentage'
    )
    melted_stats['Metric'] = melted_stats['Metric'].map({
        'Mean_Attendance': 'Average Attendance (%)', 
        'Low_Band_Percentage': 'Lectures in Low Band (%)'
    })
    
    fig_bar = px.bar(
        melted_stats, 
        y='Subject', 
        x='Percentage', 
        color='Metric', 
        orientation='h', 
        barmode='group',
        text_auto=".1f",
        color_discrete_sequence=['#1f77b4', '#d62728'], # Blue for Avg, Red for Low Band %
        height=max(400, len(selected_subjects) * 60)
    )
    
    # Lock the y-axis order to our sorted list
    fig_bar.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray': ordered_subjects},
        xaxis_title="Percentage (%)",
        yaxis_title="",
        legend_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 4. Box Plot (Distribution & Outliers)
    st.divider()
    st.markdown("### 📦 Attendance Distribution per Subject")
    st.markdown("This box plot shows the spread of attendance percentages for each subject, highlighting consistency and outliers.")
    
    fig_box = px.box(
        filtered_df,
        x='Attendance_Percentage',
        y='Subject',
        color='Subject',
        orientation='h',
        height=max(400, len(selected_subjects) * 50)
    )
    
    # Lock the y-axis order to match the first chart and hide the redundant legend
    fig_box.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray': ordered_subjects},
        xaxis_title="Attendance Percentage (%)",
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred while generating the analytics: {e}")
