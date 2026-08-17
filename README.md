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
Three different models were trained and evaluated on the engineered dataset:
- **Random Forest Classifier:** A baseline ensemble model providing robust splits on categorical and continuous data.
- **Logistic Regression:** A linear approach leveraging a `StandardScaler` pipeline to ensure proper convergence on continuous features.
- **XGBoost Classifier (Deployed):** A highly optimized gradient boosting model that utilizes sample weights (to handle class imbalances) and produces reliable probabilistic predictions. The XGBoost pipeline (`xgb-classifier-model.pkl`) is the active model serving the web application.

### 3. Interactive Streamlit Dashboard
The project features a sleek, multi-page web dashboard built with [Streamlit](https://streamlit.io/):
- **Welcome Page (`app.py`):** 
  - Provides a high-level overview and reads directly from the dataset to calculate and display key metrics (Total Lectures, Date Range, Overall Average Attendance).
  - Highlights actionable insights discovered during exploratory analysis.
- **Predict Page (`pages/1_Predict.py`):**
  - Features an intuitive, two-column form for users to input the context of a future lecture (e.g., Subject, Start Time, Weather, Internal Tests).
  - Dynamically encodes these inputs via the `utils.py` module, derives complex features (like rolling averages), and interfaces with the pre-trained XGBoost model.
  - Displays the predicted attendance band alongside visual probability bars (`st.progress`) for each class.

## Project Structure
```text
ds-ml-project/
├── data/
│   └── attendance_dataset-V2.csv        # The historical attendance data
├── model/
│   ├── xgb-classifier-model.pkl         # Trained XGBoost pipeline
│   ├── label_encoder.pkl                # Target variable encoder (Low, Medium, High)
│   └── subject_label_encoder.pkl        # Subject categorical encoder
├── notebooks/
│   ├── Feature-engineering.txt          # Guidelines for feature extraction
│   ├── Random-forect-classifier.ipynb   # Random Forest training & evaluation
│   ├── Logistic-regression-classifier.ipynb # Logistic Regression training & evaluation
│   └── XGBoost-classifier.ipynb         # XGBoost training & evaluation
├── pages/
│   ├── 1_Predict.py                     # Prediction dashboard page
│   ├── 2_Time_Slots.py                  # (Upcoming) Analysis of time slots
│   ├── 3_Subjects.py                    # (Upcoming) Analysis of subjects
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
