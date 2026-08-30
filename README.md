# Institute Attendance Predictor

## Overview
**Institute Attendance Predictor** is a Machine Learning project and interactive web application designed to forecast classroom attendance based on scheduling, environmental, and temporal factors. By analyzing historical lecture data, the application predicts whether attendance for a specific session will be **Low**, **Medium**, or **High**. 

This system helps educators and administrators proactively identify periods of low engagement and better understand what drives student attendance (e.g., weather conditions, time of day, impending holidays, or internal tests).

## Features & Specifications

### 1. Data Processing & Feature Engineering
- **Temporal Features:** Extracts logical date components such as `Day_of_Week`, `Week_Number`, `Day_Number_of_Semester`, and parses `Start_Time` into a 24-hour integer format.
- **Derived Indicators:** Automatically computes behavioral flags like `Is_First_Lecture_of_Day` and `Is_Afternoon`.
- **Momentum Metrics:** Computes the `Rolling_Avg_3` (the rolling average of the last 3 attendance values for a specific subject) directly from historical data to capture attendance momentum.
- **Time-Based Splitting:** Instead of random training splits, the dataset is strictly sorted chronologically to evaluate the model accurately on future (unseen) time periods.

### 2. Machine Learning Models
Four different models were trained and evaluated on the engineered dataset:
- **Random Forest Classifier:** A baseline ensemble model providing robust splits on categorical and continuous data.
- **Logistic Regression:** A linear approach leveraging a `StandardScaler` pipeline to ensure proper convergence on continuous features.
- **XGBoost Classifier:** A highly optimized gradient boosting model that utilizes sample weights (to handle class imbalances) and produces reliable probabilistic predictions for the attendance band.
- **Gradient Boosting Regressor:** A regression model trained to predict the exact numerical attendance (number of students) with a measured Mean Absolute Error of ±16 students.

### 3. Interactive Streamlit Dashboard
The project features a sleek, multi-page web dashboard built with [Streamlit](https://streamlit.io/):
- **Welcome Page (`app.py`):** 
  - Provides a high-level overview and reads directly from the dataset to calculate and display key metrics (Total Lectures, Date Range, Overall Average Attendance).
  - Highlights actionable insights discovered during exploratory analysis.
- **Predict Page (`pages/1_Predict.py`):**
  - Features an intuitive, two-column form for users to input the context of a future lecture (e.g., Subject, Start Time, Weather, Internal Tests).
  - Allows users to dynamically select their preferred classifier model (XGBoost, Random Forest, or Logistic Regression).
  - Displays a side-by-side output: the predicted categorical band (Low/Medium/High) with probability bars, and a specific numerical attendance estimate (with a confidence range) generated simultaneously by the Gradient Boosting Regressor.
- **Time Slots Analysis (`pages/2_Time_Slots.py`):** 
  - An interactive Plotly heatmap pinpointing consistently underperforming lecture slots, automatically flagged using the exact attendance thresholds extracted directly from the XGBoost model.
- **Subjects Analysis (`pages/3_Subjects.py`):** 
  - Comprehensive bar and box plots showcasing attendance distribution, spread, and low-band frequency per subject, complete with interactive filtering.

## Project Structure
```text
ds-ml-project/
├── data/
│   └── attendance_dataset_cleaned.csv    # The historical attendance data
├── model/
│   ├── XGBoost/                         # Trained XGBoost pipeline and thresholds
│   ├── Random_Forest/                   # Trained Random Forest pipeline
│   ├── Logistic_Regression/             # Trained Logistic Regression pipeline
│   └── GradientBoosting/                # Trained Regressor model
├── notebooks/
│   ├── Feature-engineering.txt          # Guidelines for feature extraction
│   ├── Random-forect-classifier.ipynb   # Random Forest training & evaluation
│   ├── Logistic-regression-classifier.ipynb # Logistic Regression training
│   ├── XGBoost-classifier.ipynb         # XGBoost training & evaluation
│   └── GardientBoosting.ipynb           # Gradient Boosting Regressor training
├── pages/
│   ├── 1_Predict.py                     # Prediction dashboard page
│   ├── 2_Time_Slots.py                  # Analytical visualization of time slots
│   ├── 3_Subjects.py                    # Analytical visualization of subjects
│   └── 4_What_if.py                     # (Upcoming) Scenario simulator
├── utils/
│   └── utils.py                         # Data encoding and inference helper functions
├── app.py                               # Main Streamlit application (Welcome Page)
└── pyproject.toml                       # Python project configuration and dependencies
```

## Tech Stack
- **Language:** Python >= 3.12
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Web Framework:** Streamlit
- **Data Visualization:** Matplotlib, Seaborn, Plotly

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atharv412/Institute-Attendance-Predictor.git
   cd Institute-Attendance-Predictor/ds-ml-project
   ```

2. **Install dependencies:**
   This project uses a standard `pyproject.toml` file. You can install the dependencies in a virtual environment using `uv`, `pip`, or your preferred package manager.
   ```bash
   pip install .
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *The application will launch in your default web browser at `http://localhost:8501`.*
