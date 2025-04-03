import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats # For Z-score potentially later, keep for now
import io # To handle BytesIO for data loading simulation
import os
import json
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Student Writing Style Overview")

st.title("🎓 Student Writing Style Overview")
st.markdown("""
Upload CSV data and select a student to view their writing style metrics and baseline analysis.
""")

# --- Exam Title Mapping (Copied from main.py) ---
EXAM_TITLE_MAPPING = {
    "25 BIC AS2 BUS P2 MOCK007": ("Business", "Humanities and Social Sciences"),
    "25 BIC AS2 CHEM P2 MOCK010": ("Chemistry", "Science, Health and Physics Education"),
    "25 BIC AS2 HIST P2 MOCK007": ("History", "Humanities and Social Sciences"),
    "25 BIC AS2 PHS P3 MOCK011": ("Physics", "Science, Health and Physics Education"),
    "25 BIC IG1 ECON P1 TEST002": ("Economics", "Humanities and Social Sciences"),
    "25 BIC IG1 HIST TEST001": ("History", "Humanities and Social Sciences"),
    "25 BIC IG1 ICT TEST002 THEORY": ("Information and Communication Technology", "Technologies"),
    "25 BIC IG2 BIO P1MOCK008": ("Biology", "Science, Health and Physics Education"),
    "25 BIC IG2 BIO P2MOCK009": ("Biology", "Science, Health and Physics Education"),
    "25 BIC IG2 BUS P2 MOCK008": ("Business", "Humanities and Social Sciences"),
    "25 BIC IG2 CHEM P2MOCK009": ("Chemistry", "Science, Health and Physics Education"),
    "25 BIC IG2 ENGL P2MOCK012": ("English Language", "English"),
    "25 BIC IG2 GEOG P2 MOCK010": ("Geography", "Humanities and Social Sciences"),
    "25 BIC IG2 HIST P2 MOCK010": ("History", "Humanities and Social Sciences"),
    "25 BIC IG2 ICT P2 MOCK009": ("Information and Communication Technology", "Technologies"),
    "25 BIC IG2 PHYS P2 MOCK009": ("Physics", "Science, Health and Physics Education"),
    "25 IEB FET T1 GR10 BSTD": ("Business", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR10 CATN Theory": ("Information and Communication Technology", "Technologies"),
    "25 IEB FET T1 GR10 CSTD": ("Consumer Studies (CSTD)", "Technologies"),
    "25 IEB FET T1 GR10 DRAMA": ("Drama", "The Arts"),
    "25 IEB FET T1 GR10 DSGN": ("Design", "The Arts"),
    "25 IEB FET T1 GR10 ECON Test 2": ("Economics", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR10 ENGHL": ("English Language", "English"),
    "25 IEB FET T1 GR10 GEOG": ("Geography", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR10 HIST": ("History", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR10 LFSC": ("Life Sciences", "Science, Health and Physics Education"),
    "25 IEB FET T1 GR10 PHSC": ("Physics", "Science, Health and Physics Education"),
    "25 IEB FET T1 GR10 TRSM": ("Tourism", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR10 VSLA": ("Visual Arts", "The Arts"),
    "25 IEB FET T1 GR11 BSTD": ("Business", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR11 CATN Theory": ("Information and Communication Technology", "Technologies"),
    "25 IEB FET T1 GR11 CSTD": ("Consumer Studies (CSTD)", "Technologies"),
    "25 IEB FET T1 GR11 DRAMA": ("Drama", "The Arts"),
    "25 IEB FET T1 GR11 DSGN": ("Design", "The Arts"),
    "25 IEB FET T1 GR11 ECON Test 2": ("Economics", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR11 ENGHL": ("English Language", "English"),
    "25 IEB FET T1 GR11 GEOG": ("Geography", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR11 HIST": ("History", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR11 LFSC": ("Life Sciences", "Science, Health and Physics Education"),
    "25 IEB FET T1 GR11 PHSC": ("Physics", "Science, Health and Physics Education"),
    "25 IEB FET T1 GR11 TRSM": ("Tourism", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR11 VSLA": ("Visual Arts", "The Arts"),
    "25 IEB FET T1 GR12 BSTD": ("Business", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR12 CATN Theory": ("Information and Communication Technology", "Technologies"),
    "25 IEB FET T1 GR12 CSTD": ("Consumer Studies (CSTD)", "Technologies"),
    "25 IEB FET T1 GR12 DRAMA": ("Drama", "The Arts"),
    "25 IEB FET T1 GR12 DSGN Test 1": ("Design", "The Arts"),
    "25 IEB FET T1 GR12 DSGN Test 2": ("Design", "The Arts"),
    "25 IEB FET T1 GR12 ECON Test 2": ("Economics", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR12 ENGHL Test": ("English Language", "English"),
    "25 IEB FET T1 GR12 GEOG": ("Geography", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR12 HIST P1": ("History", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR12 HIST P2": ("History", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR12 INFT THEORY": ("Information and Communication Technology", "Technologies"),
    "25 IEB FET T1 GR12 LFSC": ("Life Sciences", "Science, Health and Physics Education"),
    "25 IEB FET T1 GR12 PHSC": ("Physics", "Science, Health and Physics Education"),
    "25 IEB FET T1 GR12 TRSM": ("Tourism", "Humanities and Social Sciences"),
    "25 IEB FET T1 GR12 VSLA": ("Visual Arts", "The Arts"),
    "25 SACAI T1 Gr.10 BSTD": ("Business", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.10 CSTD": ("Consumer Studies (CSTD)", "Technologies"),
    "25 SACAI T1 Gr.10 DSGN": ("Design", "The Arts"),
    "25 SACAI T1 Gr.10 ECON": ("Economics", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.10 ENG FAL": ("English Language", "English"),
    "25 SACAI T1 Gr.10 ENG HL": ("English Language", "English"),
    "25 SACAI T1 Gr.10 GEOG": ("Geography", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.10 HIST": ("History", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.10 LFSC": ("Life Sciences", "Science, Health and Physics Education"),
    "25 SACAI T1 Gr.10 TRSM": ("Tourism", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.10 VSLA": ("Visual Arts", "The Arts"),
    "25 SACAI T1 Gr.11 BSTD": ("Business", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.11 CSTD": ("Consumer Studies (CSTD)", "Technologies"),
    "25 SACAI T1 Gr.11 DSGN": ("Design", "The Arts"),
    "25 SACAI T1 Gr.11 ECON": ("Economics", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.11 ENG FAL": ("English Language", "English"),
    "25 SACAI T1 Gr.11 ENG HL": ("English Language", "English"),
    "25 SACAI T1 Gr.11 GEOG": ("Geography", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.11 HIST": ("History", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.11 INFT TH": ("Information and Communication Technology", "Technologies"),
    "25 SACAI T1 Gr.11 LFSC": ("Life Sciences", "Science, Health and Physics Education"),
    "25 SACAI T1 Gr.11 TRSM": ("Tourism", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.11 VSLA": ("Visual Arts", "The Arts"),
    "25 SACAI T1 Gr.12 ACCN": ("Accounting", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.12 BSTD": ("Business", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.12 CATN TH": ("Information and Communication Technology", "Technologies"),
    "25 SACAI T1 Gr.12 CSTD": ("Consumer Studies (CSTD)", "Technologies"),
    "25 SACAI T1 Gr.12 DRAMA": ("Drama", "The Arts"),
    "25 SACAI T1 Gr.12 DSGN": ("Design", "The Arts"),
    "25 SACAI T1 Gr.12 ECON": ("Economics", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.12 ENG FAL": ("English Language", "English"),
    "25 SACAI T1 Gr.12 ENG HL": ("English Language", "English"),
    "25 SACAI T1 Gr.12 GEOG": ("Geography", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.12 HIST": ("History", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.12 LFSC": ("Life Sciences", "Science, Health and Physics Education"),
    "25 SACAI T1 Gr.12 TRSM": ("Tourism", "Humanities and Social Sciences"),
    "25 SACAI T1 Gr.12 VSLA": ("Visual Arts", "The Arts"),
    "G08EMS Cycle Test Term 1: P1 Question Paper [70 mins]": ("Economics", "Humanities and Social Sciences"),
    "G08EMS LS Cycle Test Term 1: P1 Question Paper [70 mins]": ("Economics", "Humanities and Social Sciences"),
    "G08ENG Cycle Test Term 1: P1 Question paper [130 mins]": ("English Language", "English"),
    "G08ENG LS Cycle Test Term 1: P1 Question paper [130 mins]": ("English Language", "English"),
    "G09ENG Cycle test Term 1: P1 Question paper: Response to texts [130 mins]": ("English Language", "English"),
    "G09ENG LS Cycle test Term 1: P1 Question paper: Response to texts [130 mins]": ("English Language", "English"),
    "G10ACC Cycle Test Term 1: P1 Question Paper [100 mins]": ("Accounting", "Humanities and Social Sciences"),
    "G10ACC LS Cycle Test Term 1: P1 Question Paper [100 mins]": ("Accounting", "Humanities and Social Sciences"),
    "G10PHS Cycle Test Term 1: P1 Question Paper [100 mins]": ("Physics", "Science, Health and Physics Education"),
    "G10PHS LS Cycle Test Term 1: P1 Question Paper [100 mins]": ("Physics", "Science, Health and Physics Education"),
    "G11PHS Cycle test Term 1: P1 Question paper [130 mins]": ("Physics", "Science, Health and Physics Education"),
    "G11PHS LS Cycle test Term 1: P1 Question paper [130 mins]": ("Physics", "Science, Health and Physics Education"),
    "G12HIS Cycle test Term 1: P1 question paper [130 mins]": ("History", "Humanities and Social Sciences"),
    "G12HIS LS Cycle test Term 1: P1 question paper [130 mins]": ("History", "Humanities and Social Sciences"),
    "G12PHS Cycle test Term 1: P1 question paper [130 mins]": ("Physics", "Science, Health and Physics Education"),
    "G12PHS LS Cycle test Term 1: P1 question paper [130 mins]": ("Physics", "Science, Health and Physics Education"),
    "G12VRT Cycle test Term 1: P1 question paper [100 mins]": ("Visual Arts", "The Arts"),
    "G12VRT LS Cycle test Term 1: P1 question paper [100 mins]": ("Visual Arts", "The Arts"),
    "Navitas Web demo": ("N/A (Unclear Subject)", "N/A"),
    "Principles of Business 11_2025_D1": ("Business", "Humanities and Social Sciences")
}

# --- Functions (Adapted from main.py) ---

# Function to add subject and category based on mapping
def add_subject_info(df, mapping):
    """Adds 'Subject' and 'Subject Category' based on 'exam_title' using the provided mapping."""
    if 'exam_title' in df.columns:
        df['exam_title'] = df['exam_title'].astype(str)
        df['Subject'] = df['exam_title'].map(lambda x: mapping.get(x, ("Unknown", "Unknown"))[0])
        df['Subject Category'] = df['exam_title'].map(lambda x: mapping.get(x, ("Unknown", "Unknown"))[1])
    else:
        st.error("'exam_title' column not found in uploaded data. Cannot map subjects.")
        df['Subject'] = "Unknown"
        df['Subject Category'] = "Unknown"
    return df

# Cache data loading to improve performance
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(uploaded_file):
    """Loads data from the uploaded CSV file, performs basic checks, conversions, and adds subject info."""
    try:
        df = pd.read_csv(uploaded_file)

        # --- Basic Column Checks ---
        # Define columns expected based on user description
        expected_cols = [
            'student_number', 'exam_id', 'exam_title',
            'max_ai_score_of_all_questions', 'average_ai_score_across_all_questions',
            'word_count', 'type_token_ratio', 'hapax_legomenon_rate', 'average_word_length',
            'contraction_ratio', 'punctuation_ratio', 'stopword_ratio', 'adverb_ratio',
            'bigram_uniqueness', 'trigram_uniqueness', 'syntax_variety', 'average_sentence_length',
            'complex_verb_ratio', 'sophisticated_adjective_ratio', 'complex_sentence_ratio',
            'burrows_delta', 'alternative_burrows_delta', 'novel_words_count'
            # Add other necessary identifiers/metadata if needed later
             'total_exams_subsmitted_by_student', 'student_exam_submission_id',
            'number_exam_questions_analysed_for_ai_content', 'id', 'exam_result_id',
             'aggregated_counts', 'created_at', 'updated_at'
        ]
        # Check for a minimal essential set for this app
        required_cols_for_app = ['student_number', 'exam_id', 'exam_title',
                                 'max_ai_score_of_all_questions', 'word_count']
        missing_required = [col for col in required_cols_for_app if col not in df.columns]
        if missing_required:
            st.error(f"Missing essential columns: {', '.join(missing_required)}. Please ensure your CSV has the correct headers.")
            return None

        # --- Add Subject Info ---
        df = add_subject_info(df, EXAM_TITLE_MAPPING)

        # --- Define Metrics ---
        # Identify numeric metrics available in the file based on expected list
        all_potential_metrics = [
            'word_count', 'type_token_ratio', 'hapax_legomenon_rate', 'average_word_length',
            'contraction_ratio', 'punctuation_ratio', 'stopword_ratio', 'adverb_ratio',
            'bigram_uniqueness', 'trigram_uniqueness', 'syntax_variety', 'average_sentence_length',
            'complex_verb_ratio', 'sophisticated_adjective_ratio', 'complex_sentence_ratio',
            'burrows_delta', 'alternative_burrows_delta', 'novel_words_count',
            'max_ai_score_of_all_questions', 'average_ai_score_across_all_questions' # Include AI scores if needed as metrics too
        ]
        available_metrics = [m for m in all_potential_metrics if m in df.columns]

        # --- Data Type Conversions ---
        for col in available_metrics:
             if col in df.columns: # Double check
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert IDs to string
        id_cols = ['student_number', 'exam_id', 'student_exam_submission_id']
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Convert dates if they exist
        datetime_cols = ['created_at', 'updated_at']
        for col in datetime_cols:
            if col in df.columns:
                 df[col] = pd.to_datetime(df[col], errors='coerce')

        st.success("Data loaded and preprocessed successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# --- Define Metrics to Analyze (based on user-provided list) ---
METRICS_TO_CALC = [
    'type_token_ratio', 'hapax_legomenon_rate', 'average_word_length',
    'punctuation_ratio', 'stopword_ratio', 'adverb_ratio',
    'bigram_uniqueness', 'trigram_uniqueness', 'syntax_variety', 'average_sentence_length',
    'complex_verb_ratio', 'sophisticated_adjective_ratio', 'complex_sentence_ratio',
    'burrows_delta', 'alternative_burrows_delta', 'novel_words_count'
]

# Function to calculate baseline stats (Simplified version for this app)
# Cache baseline calculation per student
@st.cache_data(ttl=600) # Cache for 10 mins
def calculate_student_baseline(student_data, metrics_list, ai_col, ai_thresh, wc_col, wc_thresh):
    """Calculates baseline statistics for a single student's data based on criteria."""
    baseline_submissions = student_data[
        (student_data[ai_col] <= ai_thresh) &
        (student_data[wc_col] >= wc_thresh)
    ].copy()

    if baseline_submissions.empty:
        return pd.DataFrame(), 0 # Return empty DataFrame and 0 count

    baseline_stats = {}
    for metric in metrics_list:
        if metric in baseline_submissions.columns and pd.api.types.is_numeric_dtype(baseline_submissions[metric]) and not baseline_submissions[metric].isnull().all():
            data = baseline_submissions[metric].dropna()
            if len(data) >= 1: # Need at least one point for min/max/mean
                stats = {
                    'count': len(data),
                    'mean': data.mean(),
                    'min': data.min(),
                    'p20': data.quantile(0.20) if len(data) >= 1 else np.nan, # Pandas handles small N for quantile
                    'p80': data.quantile(0.80) if len(data) >= 1 else np.nan,
                    'max': data.max(),
                    'std': data.std() if len(data) >= 2 else 0.0 # Std needs N>=2, return 0 if N=1 or std=0
                }
                baseline_stats[metric] = stats
            else:
                 baseline_stats[metric] = {k: np.nan for k in ['count', 'mean', 'std', 'min', 'p20', 'p80', 'max']}
                 baseline_stats[metric]['count'] = 0


    # Convert dict to DataFrame
    baseline_df = pd.DataFrame.from_dict(baseline_stats, orient='index')
    # Ensure all stat columns exist even if a metric had no data
    all_stat_cols = ['count', 'mean', 'std', 'min', 'p20', 'p80', 'max']
    for stat_col in all_stat_cols:
        if stat_col not in baseline_df.columns:
             baseline_df[stat_col] = np.nan
    
    baseline_df = baseline_df[all_stat_cols] # Ensure order
    baseline_df.index.name = 'Metric'
    
    return baseline_df.reset_index(), len(baseline_submissions) # Return stats DF and count of baseline exams


# --- Initialize State ---
df = None
all_student_numbers = []
selected_student = None

# Initialize session state for exam options and z-score threshold
if 'student_exam_options' not in st.session_state:
    st.session_state.student_exam_options = ["(Select an Exam)"]
if 'prev_selected_student' not in st.session_state:
    st.session_state.prev_selected_student = None
if 'z_score_threshold' not in st.session_state:
    st.session_state.z_score_threshold = 1.0  # Default z-score threshold

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("1. Upload CSV Data", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            if 'student_number' in df.columns:
                all_student_numbers = sorted(df['student_number'].astype(str).unique().tolist())
            else:
                st.error("Fatal: 'student_number' column not found in the uploaded file.")
                df = None # Prevent further processing

    # --- Student Selection ---
    st.subheader("2. Select Student")
    selected_student = st.selectbox(
        "Select Student Number",
        options=["(Select a Student)"] + all_student_numbers,
        index=0,
        disabled=(df is None)
    )
    
    # --- Check if student selection changed ---
    if selected_student != st.session_state.prev_selected_student and selected_student != "(Select a Student)" and df is not None:
        # Update exam options based on the new student selection
        student_df_temp = df[df['student_number'] == selected_student]
        if not student_df_temp.empty:
            student_exams_temp = student_df_temp[['exam_id', 'exam_title']].drop_duplicates().sort_values('exam_title')
            st.session_state.student_exam_options = ["(Select an Exam)"] + [
                f"{row['exam_title']} ({row['exam_id']})" for index, row in student_exams_temp.iterrows()
            ]
        else:
            st.session_state.student_exam_options = ["(Select an Exam)"]
        
        # Update the previous selection
        st.session_state.prev_selected_student = selected_student

    # --- Baseline Criteria ---
    st.subheader("3. Baseline Criteria")

    # Define available AI columns
    max_ai_col = 'max_ai_score_of_all_questions'
    avg_ai_col = 'average_ai_score_across_all_questions'
    
    # Check which AI score columns are available
    max_ai_available = df is not None and max_ai_col in df.columns
    avg_ai_available = df is not None and avg_ai_col in df.columns
    
    # Default thresholds and ranges
    min_val_ai = 0.0
    max_val_ai = 1.0
    default_low_threshold = 0.20

    # --- Max AI Score Threshold Slider ---
    max_ai_threshold = st.sidebar.slider(
        label=f"Max AI Score Threshold",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="Submissions with Max AI Score above this threshold will be highlighted"
    )

    # --- Average AI Score Threshold Slider ---
    avg_ai_threshold = st.sidebar.slider(
        label=f"Average AI Score Threshold",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Submissions with Average AI Score above this threshold will be highlighted"
    )

    # Minimum Word Count Slider
    min_wc = 0
    max_wc = 1000 # Default max
    default_min_wc = 50
    word_count_col = 'word_count'
    baseline_wc_disabled = True
    if df is not None and word_count_col in df.columns and pd.api.types.is_numeric_dtype(df[word_count_col]) and not df[word_count_col].isnull().all():
        valid_wc = df[word_count_col].dropna()
        min_wc = int(valid_wc.min())
        max_wc = int(valid_wc.max())
        default_min_wc = max(min_wc, min(max_wc, default_min_wc)) # Ensure default is within bounds
        baseline_wc_disabled = False


    min_word_count_baseline = st.slider(
        "Min Word Count for Baseline",
        min_value=min_wc,
        max_value=max_wc,
        value=default_min_wc,
        step=10,
        disabled=baseline_wc_disabled,
        help="Submissions with word count at or above this value will be used for baseline calculation."
    )

    # --- Add Z-Score Threshold after Min Word Count slider ---
    # Place this after the "Min Word Count for Baseline" slider
    st.subheader("Z-Score Threshold")
    z_score_threshold = st.slider(
        "Acceptable Range (|Z-Score| ≤ value)",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.z_score_threshold,
        step=0.1,
        help="Metrics with absolute z-scores exceeding this threshold will be flagged as outside acceptable range."
    )
    st.session_state.z_score_threshold = z_score_threshold  # Update session state

    # --- Exam Selection for Comparison ---
    st.subheader("4. Select Exam for Comparison")
    # Use the session state to display the updated options
    selected_exam_id = st.selectbox(
        "Select Exam to Plot",
        options=st.session_state.student_exam_options,
        index=0,
        disabled=(selected_student is None or selected_student == "(Select a Student)")
    )

    # Add helpful note about exam selection
    if selected_student != "(Select a Student)" and len(st.session_state.student_exam_options) <= 1:
        st.caption("No exams found for this student or still loading. Try selecting the student again.")

# --- Main Panel ---
if df is None:
    st.info("👈 Upload a CSV file using the sidebar to begin.")
elif selected_student is None or selected_student == "(Select a Student)":
    st.info("👈 Select a student from the sidebar to view their overview.")
else:
    # Filter data for the selected student
    student_df = df[df['student_number'] == selected_student].copy()

    if student_df.empty:
        st.error(f"No data found for student {selected_student}.")
    else:
        st.header(f"Overview for Student: {selected_student}")

        # --- 1. All Submissions Table ---
        st.subheader("All Submissions")
        # Define columns to show - include metrics, AI scores, identifiers
        # Define the desired order explicitly
        ordered_cols = ['exam_id', 'exam_title', 'Subject'] + \
                       [col for col in [max_ai_col, avg_ai_col] if col in student_df.columns] + \
                       [col for col in METRICS_TO_CALC if col in student_df.columns and col not in [max_ai_col, avg_ai_col]]
        # Ensure only existing columns are selected
        cols_to_show_all = [c for c in ordered_cols if c in student_df.columns]
        
        # Determine sort column and sort the DataFrame *before* selecting columns
        sort_col = 'created_at' if 'created_at' in student_df.columns else 'exam_id'
        # Make sure the sort column actually exists before sorting
        if sort_col in student_df.columns:
             student_df_sorted = student_df.sort_values(by=sort_col)
        else:
             # Fallback if neither 'created_at' nor 'exam_id' are present (shouldn't happen based on load_data checks)
             student_df_sorted = student_df 
             st.warning(f"Could not find default sort column ('{sort_col}'). Table may not be sorted.")

        # Apply styling to highlight AI scores above thresholds
        def highlight_high_ai_scores(df):
            # Create a copy of the dataframe styling
            style_df = df.style
            
            # Apply highlighting to max AI score column if it exists
            if max_ai_col in df.columns:
                style_df = style_df.map(
                    lambda val: 'font-weight: bold; color: red' if not pd.isna(val) and val > max_ai_threshold else '',
                    subset=[max_ai_col]
                )
            
            # Apply highlighting to avg AI score column if it exists
            if avg_ai_col in df.columns:
                style_df = style_df.map(
                    lambda val: 'font-weight: bold; color: red' if not pd.isna(val) and val > avg_ai_threshold else '',
                    subset=[avg_ai_col]
                )
            
            return style_df
        
        # Display the styled DataFrame
        st.dataframe(highlight_high_ai_scores(student_df_sorted[cols_to_show_all]))

        # --- Update Exam Selection Dropdown ---
        # Get exams for this specific student
        student_exams = student_df[['exam_id', 'exam_title']].drop_duplicates().sort_values('exam_title')
        # Update session state with the latest exam options to ensure it's current
        st.session_state.student_exam_options = ["(Select an Exam)"] + [
            f"{row['exam_title']} ({row['exam_id']})" for index, row in student_exams.iterrows()
        ]
        
        # Find the actual exam_id if a selection other than default is made
        exam_id_to_plot = None
        if selected_exam_id != "(Select an Exam)":
             try:
                 # Extract exam_id from the formatted string "(Exam Title) (exam_id)"
                 exam_id_to_plot = selected_exam_id.split('(')[-1].split(')')[0]
             except IndexError:
                 st.warning("Could not parse selected exam ID.")


        # --- 2. Baseline Calculation & Display ---
        st.subheader("Baseline Analysis")
        
        # Dynamically build baseline criteria explanation
        baseline_explanation = "Baseline calculated using submissions where:\n"
        if max_ai_col in student_df.columns:
            baseline_explanation += f"*   `{max_ai_col}` ≤ {max_ai_threshold:.2f}\n"
        if avg_ai_col in student_df.columns:
            baseline_explanation += f"*   `{avg_ai_col}` ≤ {avg_ai_threshold:.2f}\n"
        if word_count_col in student_df.columns:
            baseline_explanation += f"*   `{word_count_col}` ≥ {min_word_count_baseline}\n"
        
        st.markdown(baseline_explanation)

        # Filter for baseline submissions *for this student*
        baseline_criteria = []
        
        # Add max AI score criteria if column exists
        if max_ai_col in student_df.columns:
            baseline_criteria.append(student_df[max_ai_col] <= max_ai_threshold)
        
        # Add average AI score criteria if column exists
        if avg_ai_col in student_df.columns:
            baseline_criteria.append(student_df[avg_ai_col] <= avg_ai_threshold)
        
        # Add word count criteria
        if word_count_col in student_df.columns:
            baseline_criteria.append(student_df[word_count_col] >= min_word_count_baseline)
        
        # Combine all criteria with logical AND
        if baseline_criteria:
            baseline_criteria_met = baseline_criteria[0]
            for criteria in baseline_criteria[1:]:
                baseline_criteria_met = baseline_criteria_met & criteria
        else:
            baseline_criteria_met = pd.Series(True, index=student_df.index)
            
        baseline_submissions_df = student_df[baseline_criteria_met]
        num_baseline_exams = len(baseline_submissions_df)

        if num_baseline_exams == 0:
            st.warning("No submissions found matching the baseline criteria for this student.")
        else:
            st.info(f"Calculating baseline statistics using **{num_baseline_exams}** submission(s).")

            # Calculate stats (can reuse the function or do a simpler agg here)
            metrics_in_data = [m for m in METRICS_TO_CALC if m in student_df.columns and pd.api.types.is_numeric_dtype(student_df[m])]
            
            if not metrics_in_data:
                 st.warning("No numeric metrics found in the data to calculate baselines for.")
            else:
                # Manually calculate statistics for each metric
                # Create a new DataFrame to hold the results
                baseline_stats_data = []
                metrics_to_agg = [m for m in metrics_in_data if m in baseline_submissions_df.columns and pd.api.types.is_numeric_dtype(baseline_submissions_df[m]) and not baseline_submissions_df[m].isnull().all()]
                
                if not metrics_to_agg:
                    st.warning("No metrics with valid data found in the baseline submissions.")
                    baseline_stats_df = pd.DataFrame() # Assign empty DF
                else:
                    try:
                        # Calculate statistics for each metric manually
                        for metric in metrics_to_agg:
                            values = baseline_submissions_df[metric].dropna()
                            count = len(values)
                            
                            if count > 0:
                                # Calculate statistics
                                mean_val = values.mean()
                                min_val = values.min()
                                max_val = values.max()
                                p20_val = values.quantile(0.20) if count >= 1 else np.nan
                                p80_val = values.quantile(0.80) if count >= 1 else np.nan
                                # For std, use count >= 2 check
                                std_val = values.std() if count >= 2 else 0.0
                                
                                # Add to results
                                baseline_stats_data.append({
                                    'Metric': metric,
                                    'count': count,
                                    'mean': mean_val,
                                    'std': std_val,
                                    'min': min_val,
                                    'p20': p20_val,
                                    'p80': p80_val,
                                    'max': max_val
                                })
                        
                        # Create DataFrame from manually calculated statistics
                        if baseline_stats_data:
                            baseline_stats_df = pd.DataFrame(baseline_stats_data)
                            baseline_stats_df.set_index('Metric', inplace=True)
                        else:
                            baseline_stats_df = pd.DataFrame() # Empty if no stats calculated
                            
                    except Exception as e:
                        st.error(f"Error during baseline aggregation: {e}")
                        st.write("Baseline Submissions Data Slice (for metrics attempted):")
                        st.dataframe(baseline_submissions_df[metrics_to_agg].head())
                        baseline_stats_df = pd.DataFrame() # Assign empty DF on error

                # Display Baseline Statistics Table
                if not baseline_stats_df.empty:
                    st.dataframe(baseline_stats_df.style.format("{:.2f}", na_rep="N/A"))

                    # --- 3. Baseline Graphs (3 per row) ---
                    st.subheader("Baseline Metric Distributions")

                    # Get data for the selected exam if specified
                    exam_comparison_data = None
                    metrics_outside_range = 0  # Counter for metrics outside acceptable range
                    flagged_metrics_data = []  # To collect metrics outside range for summary table
                    
                    if exam_id_to_plot:
                        exam_comparison_data = student_df[student_df['exam_id'] == exam_id_to_plot].iloc[0] if not student_df[student_df['exam_id'] == exam_id_to_plot].empty else None
                        if exam_comparison_data is None:
                            st.warning(f"Could not find data for selected comparison exam ID: {exam_id_to_plot}")

                    # Process metrics in batches of 3 (for 3 columns)
                    metrics_per_row = 3
                    total_metrics = len(metrics_to_agg)
                    
                    # Process each row of metrics
                    for row_start in range(0, total_metrics, metrics_per_row):
                        # Get metrics for this row (up to 3)
                        row_metrics = metrics_to_agg[row_start:row_start + metrics_per_row]
                        
                        # Create columns for this row
                        cols = st.columns(metrics_per_row)
                        
                        # Process each metric in this row
                        for idx, metric in enumerate(row_metrics):
                            with cols[idx]:
                                # Create figure
                                fig = go.Figure()
                                fig.add_trace(go.Box(
                                    y=baseline_submissions_df[metric].dropna(),
                                    name=f'{metric} Baseline (N={int(baseline_stats_df.loc[metric, "count"])})',
                                    boxpoints='all',
                                    jitter=0.3,
                                    pointpos=-1.8,
                                    marker_color='lightblue',
                                    line_color='blue'
                                ))
                                
                                # Add comparison point if exam selected and metric exists
                                metric_outside_range = False
                                z_score = None
                                
                                if exam_comparison_data is not None and metric in exam_comparison_data and not pd.isna(exam_comparison_data[metric]):
                                    # Calculate z-score if we have std dev
                                    if baseline_stats_df.loc[metric, 'count'] >= 2 and baseline_stats_df.loc[metric, 'std'] > 0:
                                        z_score = (exam_comparison_data[metric] - baseline_stats_df.loc[metric, 'mean']) / baseline_stats_df.loc[metric, 'std']
                                        metric_outside_range = abs(z_score) > z_score_threshold
                                        if metric_outside_range:
                                            metrics_outside_range += 1
                                            # Store info for summary table
                                            flagged_metrics_data.append({
                                                'Metric': metric,
                                                'Exam Value': exam_comparison_data[metric],
                                                'Z-Score': z_score,
                                                'Baseline Mean': baseline_stats_df.loc[metric, 'mean'],
                                                'Baseline Std': baseline_stats_df.loc[metric, 'std'],
                                                'Baseline Min': baseline_stats_df.loc[metric, 'min'],
                                                'Baseline Max': baseline_stats_df.loc[metric, 'max']
                                            })
                                    
                                    # Add the comparison point
                                    fig.add_trace(go.Scatter(
                                        x=[f'{metric} Baseline (N={int(baseline_stats_df.loc[metric, "count"])})'],
                                        y=[exam_comparison_data[metric]],
                                        mode='markers',
                                        name=f'Exam {exam_id_to_plot}',
                                        marker=dict(
                                            color='red', 
                                            size=12, 
                                            symbol='star',
                                            line=dict(width=2, color='darkred') if metric_outside_range else dict(width=1, color='red')
                                        ),
                                        hoverinfo='text',
                                        text=f'Exam {exam_id_to_plot}<br>{metric}: {exam_comparison_data[metric]:.2f}' + 
                                              (f'<br>Z-score: {z_score:.2f}' if z_score is not None else '')
                                    ))

                                fig.update_layout(
                                    title=f'{metric}',
                                    yaxis_title="Value",
                                    xaxis_title="",
                                    showlegend=False,
                                    height=400,
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                
                                # Display the plot
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add warning if metric is outside range
                                if metric_outside_range:
                                    st.markdown(f"<span style='color:red; font-weight:bold;'>⚠️ Warning: Metric outside acceptable range (|z-score| = {abs(z_score):.2f} > {z_score_threshold})</span>", unsafe_allow_html=True)
                                elif z_score is not None:
                                    st.markdown(f"Z-score: {z_score:.2f}")
                        
                        # Add spacing between rows
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Add summary of metrics outside acceptable range if comparison exam selected
                    if exam_comparison_data is not None:
                        st.markdown("---")
                        st.subheader("Summary")
                        
                        # Calculate the total number of metrics that had valid z-scores
                        total_valid_metrics = sum(1 for metric in metrics_to_agg if 
                                               metric in exam_comparison_data and 
                                               not pd.isna(exam_comparison_data[metric]) and
                                               baseline_stats_df.loc[metric, 'count'] >= 2 and 
                                               baseline_stats_df.loc[metric, 'std'] > 0)
                        
                        # Display ratio of metrics outside range
                        if metrics_outside_range > 0:
                            st.markdown(f"<span style='color:red; font-weight:bold;'>⚠️ {metrics_outside_range}/{total_valid_metrics} metric(s) outside acceptable range (|z-score| > {z_score_threshold})</span>", unsafe_allow_html=True)
                            
                            # Create and display detailed table of flagged metrics
                            if flagged_metrics_data:
                                st.subheader("Flagged Metrics Detail")
                                flagged_df = pd.DataFrame(flagged_metrics_data)
                                # Set index to Metric for better display
                                flagged_df.set_index('Metric', inplace=True)
                                # Format the table for better readability
                                st.dataframe(flagged_df.style.format({
                                    'Exam Value': '{:.2f}',
                                    'Z-Score': '{:.2f}',
                                    'Baseline Mean': '{:.2f}',
                                    'Baseline Std': '{:.2f}',
                                    'Baseline Min': '{:.2f}',
                                    'Baseline Max': '{:.2f}'
                                }))
                        else:
                            st.markdown(f"✅ All metrics within acceptable range (|z-score| ≤ {z_score_threshold})")
                else:
                     st.warning("Baseline statistics could not be calculated or displayed.") 