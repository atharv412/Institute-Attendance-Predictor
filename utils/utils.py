import pandas as pd

# Hardcoded mappings derived from the dataset
SUBJECT_MAPPING = {
    "DS & ML Practical": 0,
    "Data Science & Machine Learning": 1,
    "Industry Readiness Program": 2,
    "Innovation and Entrepreneurship Development": 3,
    "MAD Practical": 4,
    "Mini Project": 5,
    "Mobile Application Development": 6,
    "Principles of Cloud Management and Security": 7,
    "STQA Practical": 8,
    "Software Testing and Quality Assurance": 9
}

FACULTY_MAPPING = {
    "DPY_AAB": 0,
    "IRP": 1,
    "JSP": 2,
    "KSD_SDM": 3,
    "MLK_SP": 4,
    "SP": 5,
    "SP_MLK": 6,
    "SSP": 7,
    "SSP_ASP": 8
}

WEATHER_MAPPING = {
    "Cloudy": 0,
    "Rainy": 1,
    "Sunny": 2
}

FACULTY_PER_SUBJECT = {
    "DS & ML Practical": "MLK_SP",
    "Data Science & Machine Learning": "SP_MLK",
    "Industry Readiness Program": "IRP",
    "Innovation and Entrepreneurship Development": "JSP",
    "MAD Practical": "SSP_ASP",
    "Mini Project": "SP",
    "Mobile Application Development": "SSP",
    "Principles of Cloud Management and Security": "KSD_SDM",
    "STQA Practical": "DPY_AAB",
    "Software Testing and Quality Assurance": "DPY_AAB"
}

DAY_MAP = {
    'Monday': 0, 
    'Tuesday': 1, 
    'Wednesday': 2, 
    'Thursday': 3, 
    'Friday': 4, 
    'Saturday': 5, 
    'Sunday': 6
}

BINARY_MAP = {'No': 0, 'Yes': 1}
HOLIDAY_MAP = {'No': 0, 'Before': 1, 'After': 2}
PRACTICAL_THEORY_MAP = {'Theory': 0, 'Practical': 1}

def extract_hour(time_str):
    time_str = time_str.strip()
    parts = time_str.split(' ')
    time_part = parts[0]
    meridian = parts[1] if len(parts) > 1 else ''
    time_part = time_part.replace('.', ':')
    hour = int(time_part.split(':')[0])
    if meridian == 'PM' and hour != 12:
        hour += 12
    if meridian == 'AM' and hour == 12:
        hour = 0
    return hour

def parse_gap(gap_str):
    if gap_str == 'Same Day':
        return 0
    else:
        return int(gap_str.split(' ')[0])

def compute_rolling_avg_3(subject, dataset_path):
    """
    Computes the rolling average of the last 3 attendance values for a specific subject 
    using the original CSV dataset.
    """
    df = pd.read_csv(dataset_path)
    # Sort chronologically by subject
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by=['Subject', 'Date', 'Lecture_Number'])
    
    # Filter for the specific subject
    subject_df = df[df['Subject'] == subject]
    
    if len(subject_df) == 0:
        return 0.0
    elif len(subject_df) < 3:
        return subject_df['Attendance_Percentage'].mean()
    else:
        # Last 3 records
        return subject_df['Attendance_Percentage'].iloc[-3:].mean()

def encode_inputs(inputs, dataset_path):
    """
    Takes user inputs dictionary and returns a properly ordered 1-row DataFrame 
    ready for XGBoost prediction.
    """
    
    # 1. Derive missing fields
    faculty = FACULTY_PER_SUBJECT.get(inputs['subject'], "")
    # Day number of semester: Since semester starts on Mon 22-Jun-2026, 
    # it's just (Week_Number - 1) * 7 + Day_of_Week
    day_of_week_num = DAY_MAP[inputs['day_of_week']]
    day_number_of_semester = ((inputs['week_number'] - 1) * 7) + day_of_week_num + 1
    
    # Time based derivations
    start_time_hour = extract_hour(inputs['start_time'])
    is_first_lecture = 1 if inputs['lecture_number'] == 1 else 0
    is_afternoon = 1 if start_time_hour >= 12 else 0
    
    # Rolling Avg
    rolling_avg_3 = compute_rolling_avg_3(inputs['subject'], dataset_path)
    
    # 2. Encode categorical fields
    encoded = {
        'Day_of_Week': day_of_week_num,
        'Lecture_Number': inputs['lecture_number'],
        'Start_Time': start_time_hour,
        'Subject': SUBJECT_MAPPING[inputs['subject']],
        'Faculty_ID': FACULTY_MAPPING[faculty],
        'Previous_Lecture_Attendance': inputs['previous_lecture_attendance'],
        'Gap_Since_Previous_Lecture': parse_gap(inputs['gap_since_previous']),
        'Practical_Theory': PRACTICAL_THEORY_MAP[inputs['practical_theory']],
        'Internal_Test_Week': BINARY_MAP[inputs['internal_test_week']],
        'Assignment_Due': BINARY_MAP[inputs['assignment_due']],
        'Holiday_Before_After': HOLIDAY_MAP[inputs['holiday_before_after']],
        'Weather': WEATHER_MAPPING[inputs['weather']],
        'Special_Event': BINARY_MAP[inputs['special_event']],
        'Week_Number': inputs['week_number'],
        'Day_Number_of_Semester': day_number_of_semester,
        'Rolling_Avg_3': rolling_avg_3,
        'Is_First_Lecture_of_Day': is_first_lecture,
        'Is_Afternoon': is_afternoon
    }
    
    # Important: Feature names must be in exactly the same order as in the training data
    expected_order = [
        'Day_of_Week', 'Lecture_Number', 'Start_Time', 'Subject', 'Faculty_ID',
        'Previous_Lecture_Attendance', 'Gap_Since_Previous_Lecture', 'Practical_Theory',
        'Internal_Test_Week', 'Assignment_Due', 'Holiday_Before_After', 'Weather',
        'Special_Event', 'Week_Number', 'Day_Number_of_Semester', 'Rolling_Avg_3',
        'Is_First_Lecture_of_Day', 'Is_Afternoon'
    ]
    
    # Assemble dataframe
    df_encoded = pd.DataFrame([encoded])
    df_encoded = df_encoded[expected_order] # Reorder to match model
    
    return df_encoded
