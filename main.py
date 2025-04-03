import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats # For Z-score
import io # To handle BytesIO for data loading simulation
import os
import json
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Writing Style Analytics Dashboard")

st.title("📊 Writing Style Analytics Dashboard")
st.markdown("""
Upload your CSV data with the specified column names to analyze writing style metrics.
This tool helps visualize how metrics for potentially AI-assisted submissions
deviate from a student's typical baseline (derived from their low-AI score submissions).
""")

# --- Exam Title Mapping ---
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

# --- Functions ---

# Function to add subject and category based on mapping
def add_subject_info(df, mapping):
    """Adds 'Subject' and 'Subject Category' based on 'exam_title' using the provided mapping."""
    # Ensure 'exam_title' is string type for reliable mapping
    if 'exam_title' in df.columns:
        df['exam_title'] = df['exam_title'].astype(str)
        # Map using the dictionary, providing default values for titles not found
        df['Subject'] = df['exam_title'].map(lambda x: mapping.get(x, ("Unknown", "Unknown"))[0])
        df['Subject Category'] = df['exam_title'].map(lambda x: mapping.get(x, ("Unknown", "Unknown"))[1])
    else:
        # Handle case where exam_title column is missing
        st.error("'exam_title' column not found in uploaded data. Cannot map subjects.")
        df['Subject'] = "Unknown"
        df['Subject Category'] = "Unknown"
    return df

# Cache data loading to improve performance
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(uploaded_file):
    """Loads data from the uploaded CSV file and adds subject info."""
    try:
        df = pd.read_csv(uploaded_file)

        # --- Basic Column Checks ---
        required_cols_for_analysis = [
            'student_number', 'exam_id', 'exam_title', 'max_ai_score_of_all_questions',
            'word_count'
        ]
        missing_required = [col for col in required_cols_for_analysis if col not in df.columns]
        if missing_required:
            st.error(f"Missing essential columns: {', '.join(missing_required)}. Please ensure your CSV has the correct headers.")
            return None

        # --- Add Subject Info Early ---
        df = add_subject_info(df, EXAM_TITLE_MAPPING) # Apply mapping

        # --- Data Type Conversions ---
        numeric_metrics_and_scores = [
            'max_ai_score_of_all_questions', 'average_ai_score_across_all_questions', 'word_count',
            'type_token_ratio', 'hapax_legomenon_rate', 'average_word_length', 'contraction_ratio',
            'punctuation_ratio', 'stopword_ratio', 'adverb_ratio', 'bigram_uniqueness',
            'trigram_uniqueness', 'syntax_variety', 'average_sentence_length',
            'complex_verb_ratio', 'sophisticated_adjective_ratio', 'complex_sentence_ratio',
            'burrows_delta', 'alternative_burrows_delta', 'novel_words_count',
            'total_exams_subsmitted_by_student', 'number_exam_questions_analysed_for_ai_content',
            'id', 'exam_result_id'
        ]
        for col in numeric_metrics_and_scores:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        id_cols = ['student_number', 'exam_id', 'student_exam_submission_id'] # 'exam_title' already handled
        for col in id_cols:
            if col in df.columns:
                 df[col] = df[col].astype(str)

        datetime_cols = ['created_at', 'updated_at']
        for col in datetime_cols:
             if col in df.columns:
                 df[col] = pd.to_datetime(df[col], errors='coerce')

        st.success("Data loaded and preprocessed successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# Cache baseline calculation
@st.cache_data(ttl=3600)
def calculate_baselines(df, low_ai_threshold_col, low_ai_threshold_val, metrics_to_analyze):
    """
    Calculates baseline statistics (mean, std, min, p20, p80, max, count)
    for each student using low-AI submissions.
    """
    baseline_stats = {}
    # Use the correct AI score column name for filtering
    low_ai_df = df[df[low_ai_threshold_col] <= low_ai_threshold_val].copy()

    if low_ai_df.empty:
        st.warning(f"No submissions found with '{low_ai_threshold_col}' <= {low_ai_threshold_val} to calculate baselines.")
        return baseline_stats

    # Group by student and calculate stats for each metric
    grouped = low_ai_df.groupby('student_number')
    calculated_metrics_count = 0
    for metric in metrics_to_analyze:
        if metric in low_ai_df.columns and not low_ai_df[metric].isnull().all():
            # Define aggregations including percentiles
            # Note: Quantile calculation might require specific handling for small groups
            agg_funcs = ['mean', 'std', 'count', 'min', 'max']

            # Add quantiles safely
            def q20(x): return x.quantile(0.20) if len(x.dropna()) >= 1 else np.nan # Need at least 1 for quantile
            def q80(x): return x.quantile(0.80) if len(x.dropna()) >= 1 else np.nan

            # Check group sizes before applying quantile aggregations if needed (though agg might handle it)
            # Pandas agg generally handles this, but custom functions need care.
            stats = grouped[metric].agg(agg_funcs + [q20, q80]).reset_index()

            # Rename columns for clarity
            stats.rename(columns={
                'mean': f'{metric}_mean', 'std': f'{metric}_std',
                'count': f'{metric}_count', 'min': f'{metric}_min',
                'max': f'{metric}_max', 'q20': f'{metric}_p20',
                'q80': f'{metric}_p80'
                }, inplace=True)


            # Store stats keyed by student number for easy lookup
            for _, row in stats.iterrows():
                student = row['student_number']
                if student not in baseline_stats:
                    baseline_stats[student] = {}

                # Store calculated stats
                baseline_stats[student][f'{metric}_count'] = row[f'{metric}_count']
                baseline_stats[student][f'{metric}_mean'] = row[f'{metric}_mean']
                baseline_stats[student][f'{metric}_min'] = row[f'{metric}_min']
                baseline_stats[student][f'{metric}_p20'] = row[f'{metric}_p20']
                baseline_stats[student][f'{metric}_p80'] = row[f'{metric}_p80']
                baseline_stats[student][f'{metric}_max'] = row[f'{metric}_max']

                # Handle std: NaN if count < 2, 0 if count >= 2 but std is 0 (all values same)
                if row[f'{metric}_count'] < 2:
                    baseline_stats[student][f'{metric}_std'] = np.nan # Not enough data points for std dev
                elif pd.isna(row[f'{metric}_std']):
                     baseline_stats[student][f'{metric}_std'] = 0.0
                else:
                    baseline_stats[student][f'{metric}_std'] = row[f'{metric}_std']
            calculated_metrics_count += 1

    if calculated_metrics_count == 0:
        st.warning("No valid metrics found with data in the low-AI baseline submissions.")

    return baseline_stats


def add_ai_category(df, score_col, low_thresh, high_thresh):
    """Adds a category column based on the specified AI score column."""
    conditions = [
        df[score_col] <= low_thresh,
        df[score_col] >= high_thresh,
    ]
    choices = ['Low (Baseline)', 'High (Target)']
    df['ai_category'] = np.select(conditions, choices, default='Mid (Ignored)')
    return df

def calculate_deviation(row, baseline_stats, metric, method='z_score'):
    """Calculates deviation of a row's metric from its student's baseline."""
    student = row['student_number']
    metric_mean_key = f'{metric}_mean'
    metric_std_key = f'{metric}_std'

    # Check if baseline exists for this student and metric
    if student not in baseline_stats or metric_mean_key not in baseline_stats[student]:
        return np.nan

    baseline_mean = baseline_stats[student][metric_mean_key]
    baseline_std = baseline_stats[student].get(metric_std_key, np.nan) # Use .get for safety
    current_value = row[metric]

    # Check for NaNs in essential values
    if pd.isna(current_value) or pd.isna(baseline_mean):
         return np.nan

    # Convert to float safely, handling potential non-numeric values gracefully
    try:
        current_value = float(current_value)
        baseline_mean = float(baseline_mean)
        # Only convert std if it's needed and not NaN
        if method == 'z_score' and not pd.isna(baseline_std):
            baseline_std = float(baseline_std)
    except (ValueError, TypeError):
        return np.nan # If conversion fails

    # Calculate deviation based on method
    if method == 'z_score':
        # Handle std dev being NaN (e.g., only 1 baseline point) or zero
        if pd.isna(baseline_std) or baseline_std == 0:
             # Return 0 if value equals mean, otherwise NaN (or infinity - NaN is safer)
             return 0.0 if current_value == baseline_mean else np.nan
        else:
            return (current_value - baseline_mean) / baseline_std
    elif method == 'raw_difference':
        return current_value - baseline_mean
    elif method == 'percent_difference':
        # Avoid division by zero or near-zero issues
        if abs(baseline_mean) > 1e-9: # Use a small threshold instead of direct zero check
             return ((current_value - baseline_mean) / baseline_mean) * 100
        elif current_value == baseline_mean:
             return 0.0 # If both are zero or very close
        else:
             return np.nan # Or np.inf / -np.inf if you prefer
    else:
        return np.nan

# --- Define Helper Functions Earlier --- NEW PLACEMENT
def get_baseline_stat(student, stat_key_suffix, baseline_stats, metric):
    """Safely retrieves a specific baseline statistic for a student and metric."""
    return baseline_stats.get(student, {}).get(f'{metric}_{stat_key_suffix}', np.nan)


# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls & Filters")

    uploaded_file = st.file_uploader("1. Upload CSV Data", type=["csv"])

    # Placeholders - these will be populated after data load
    df = None
    all_metrics = []
    all_student_numbers = []
    selected_metric = None # Initialize selected_metric here
    # Initialize lists for dynamic filters
    all_subjects = []
    all_subject_categories = []
    ai_score_col = 'max_ai_score_of_all_questions' # Default, but check if exists

    if uploaded_file is not None:
        df = load_data(uploaded_file) # This now includes subject mapping
        if df is not None:
            # Verify the main AI score column exists
            if ai_score_col not in df.columns:
                st.sidebar.error(f"Column '{ai_score_col}' not found. Please check CSV header.")
                df = None # Prevent further processing
            else:
                # Define potential metric columns
                cols_to_exclude = [
                    'student_number', 'total_exams_subsmitted_by_student', 'exam_id', 'exam_title',
                    'student_exam_submission_id', 'number_exam_questions_analysed_for_ai_content',
                    'max_ai_score_of_all_questions', 'average_ai_score_across_all_questions', 'id',
                    'exam_result_id', 'aggregated_counts', 'created_at', 'updated_at',
                    'ai_category', 'Subject', 'Subject Category', # Exclude added/non-metric columns
                ]
                potential_metrics = [col for col in df.columns if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(df[col])]
                all_metrics = sorted(potential_metrics)

                # Get initial full list of students for the dropdown default
                if 'student_number' in df.columns:
                    all_student_numbers = sorted(df['student_number'].astype(str).unique().tolist())
                else:
                    all_student_numbers = [] # Should not happen if load_data checks pass
    
    # --- MOVED: Analysis Options to top ---
    st.subheader("2. Analysis Options")
    if not all_metrics:
        st.warning("Upload data with numeric metric columns.")
    else:
         # Default to 'average_sentence_length' if available, otherwise first metric
        default_metric = 'average_sentence_length'
        if default_metric not in all_metrics:
            default_metric = all_metrics[0] if all_metrics else None

        selected_metric = st.selectbox(
            "Select Metric to Analyze",
            options=all_metrics,
            index=all_metrics.index(default_metric) if default_metric in all_metrics else 0,
            disabled=df is None
        )
        
    # If still None after the selectbox (e.g., if no metrics or all_metrics is empty)
    if selected_metric is None and all_metrics:
        selected_metric = all_metrics[0] if all_metrics else "no_metric_available"
    elif selected_metric is None:
        selected_metric = "no_metric_available"

    deviation_method = st.radio(
        "Deviation Calculation Method",
        ('z_score', 'raw_difference', 'percent_difference'),
        index=0, # Default to z_score
        help="How to measure deviation from baseline: Z-score (std devs), Raw (value - mean), Percent ((value-mean)/mean).",
        disabled=df is None
    )

    # --- AI Score Selection --- NEW
    st.subheader("3. AI Score Selection")
    # Define the available AI score columns (make sure these exist in your potential CSVs)
    available_ai_score_cols = []
    if df is not None:
        if 'max_ai_score_of_all_questions' in df.columns:
            available_ai_score_cols.append('max_ai_score_of_all_questions')
        if 'average_ai_score_across_all_questions' in df.columns:
            available_ai_score_cols.append('average_ai_score_across_all_questions')

    if not available_ai_score_cols:
        st.warning("No recognized AI score columns found (expected 'max...' or 'average...'). Thresholds disabled.")
        ai_score_col = None # No column selected
        ai_scores_present = False
    elif len(available_ai_score_cols) == 1:
        ai_score_col = available_ai_score_cols[0]
        st.caption(f"Using AI score column: `{ai_score_col}`")
    else:
        ai_score_col = st.radio(
            "Select AI Score Column for Analysis:",
            options=available_ai_score_cols,
            index=0, # Default to max_ai_score
            key='ai_score_selector',
            horizontal=True
        )

    # --- AI Score Thresholds --- (Now dependent on selected ai_score_col)
    st.subheader("4. AI Score Thresholds")
    # Check if AI score column has any non-null values before calculating min/max
    ai_scores_present = False
    min_val_ai = 0.0
    max_val_ai = 1.0
    default_low = 0.25 # User requested default
    default_high = 0.8 # User requested default

    if df is not None and ai_score_col is not None and ai_score_col in df.columns and not df[ai_score_col].isnull().all():
        ai_scores_present = True
        # Calculate min/max ignoring NaNs for the SELECTED column
        min_val_ai = float(df[ai_score_col].min(skipna=True))
        max_val_ai = float(df[ai_score_col].max(skipna=True))
        # Adjust defaults slightly if range is very small or based on actual range
        range_ai = max_val_ai - min_val_ai
        if range_ai > 1e-6: # Avoid issues if max == min
            default_low = min(0.1, min_val_ai + range_ai * 0.1)
            default_high = max(0.7, min_val_ai + range_ai * 0.7)
        else: # If max == min, set defaults to that value
            default_low = min_val_ai
            default_high = max_val_ai

        # Ensure defaults are within bounds and high > low (unless range is zero)
        default_low = max(min_val_ai, default_low)
        default_high = min(max_val_ai, default_high)
        if default_high <= default_low and range_ai > 1e-6:
             default_high = default_low + range_ai * 0.5
             default_high = min(default_high, max_val_ai)
             # Final check
        if default_high <= default_low:
                  default_high = max_val_ai # Fallback

    low_ai_threshold = st.slider(
        f"Low AI Threshold ({ai_score_col} <= value is baseline)",
        min_value=min_val_ai,
        max_value=max_val_ai,
        value=default_low,
        step=0.01,
        disabled=not ai_scores_present # Disable if no scores or no column selected
    )

    high_ai_threshold = st.slider(
        f"High AI Threshold ({ai_score_col} >= value is target)",
        min_value=min_val_ai,
        max_value=max_val_ai,
        value=default_high,
        step=0.01,
        disabled=not ai_scores_present # Disable if no scores or no column selected
    )

    # Threshold validity check will happen later, after filtering

    # --- Data Filters ---
    st.subheader("5. Data Filters")

    # Add checkbox for excluding null AI scores (refers to the selected column)
    exclude_null_ai = st.checkbox(f"Exclude submissions where '{ai_score_col}' is missing", value=True,
                                  help=f"If checked, rows with no value in the selected '{ai_score_col}' column will be removed before analysis.",
                                  disabled=df is None or not ai_scores_present or ai_score_col is None)
    
    # NEW: Raw Value Range Filter
    metric_range = None # Initialize metric_range
    metric_min = 0
    metric_max = 100
    lower_fence = 0
    upper_fence = 100
    
    if df is not None and selected_metric != "no_metric_available" and selected_metric in df.columns and not df[selected_metric].dropna().empty:
        metric_values = df[selected_metric].dropna()
        metric_min = float(metric_values.min())
        metric_max = float(metric_values.max())
        
        # Calculate box plot fences (Tukey's method)
        q1 = float(metric_values.quantile(0.25))
        q3 = float(metric_values.quantile(0.75))
        iqr = q3 - q1
        lower_fence = max(metric_min, q1 - 1.5 * iqr)  # Don't go below actual min
        upper_fence = min(metric_max, q3 + 1.5 * iqr)  # Don't go above actual max

        # Add range slider for metric value filtering only if conditions are met
        st.write(f"Filter '{selected_metric}' values:")
        metric_range = st.slider(
            f"Range of {selected_metric} values to include",
            min_value=float(metric_min),
            max_value=float(metric_max),
            value=(float(lower_fence), float(upper_fence)),
            step=max(0.01, (metric_max - metric_min) / 100), # Ensure step is positive
            help=f"Default range is set to box plot whiskers (excludes outliers). Adjust to filter {selected_metric} values.",
            key=f"metric_range_slider_{selected_metric}" # Add key for stability
        )
    elif df is not None and selected_metric != "no_metric_available": # Metric selected but might be empty or not numeric
        st.write(f"Filter '{selected_metric}' values:")
        st.slider(
            f"Range of {selected_metric} values to include",
            0.0, 1.0, (0.0, 1.0),
            disabled=True,
            help=f"Cannot determine range for '{selected_metric}'. Check data."
        )
    else: # No data or no metric selected
        st.write("Filter 'Selected Metric' values:")
        st.slider(
            "Range of values to include",
            0.0, 1.0, (0.0, 1.0),
            disabled=True,
            help="Upload data and select a metric first."
        )

    # Subject and Category Filters
    selected_subjects_user = []
    selected_categories_user = []
    if df is not None and 'Subject' in df.columns and 'Subject Category' in df.columns:
        unique_subjects_all = sorted(df['Subject'].astype(str).unique().tolist())
        unique_categories_all = sorted(df['Subject Category'].astype(str).unique().tolist())
        st.caption("Subject and Category filters will auto-update based on other filters.")
        # Define the multiselect widgets here (they will be updated later based on filtered data)
        selected_subjects_user = st.multiselect(
            "Filter by Subject",
            options=unique_subjects_all,
            default=unique_subjects_all,
            key="subject_multi_initial",
            label_visibility="collapsed"
        )
        selected_categories_user = st.multiselect(
            "Filter by Subject Category",
            options=unique_categories_all,
            default=unique_categories_all,
            key="category_multi_initial",
            label_visibility="collapsed"
        )
    else:
        st.caption("Subject/Category data not available or missing columns.")

    # Word count filter
    word_count_col = 'word_count'
    min_word_count = 0 # Initialize
    word_count_slider_disabled = True
    if df is not None and word_count_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[word_count_col]) and not df[word_count_col].isnull().all():
            max_wc = int(df[word_count_col].max(skipna=True))
            min_wc = int(df[word_count_col].min(skipna=True))
            non_null_wc = df[word_count_col].dropna()
            default_wc = 100
            if len(non_null_wc) > 10:
                calculated_default = max(min_wc, int(non_null_wc.quantile(0.10)))
                default_wc = max(default_wc, calculated_default)
            elif len(non_null_wc) > 0:
                default_wc = max(min_wc, 100)

            default_wc = min(default_wc, max_wc) 
            default_wc = max(default_wc, min_wc) 
            min_word_count = st.slider(
                f"Min Word Count per Submission ({word_count_col})",
                min_value=min_wc,
                max_value=max_wc,
                value=default_wc,
                step=10,
                help="Exclude submissions with fewer words than this."
            )
            word_count_slider_disabled = False
        else:
            st.sidebar.caption(f"'{word_count_col}' column has no numeric values. Filter disabled.")
            min_word_count = 0
    elif df is not None:
        st.sidebar.caption(f"'{word_count_col}' column not found. Filter disabled.")
        min_word_count = 0

    if word_count_slider_disabled and df is not None:
        st.slider(f"Min Word Count per Submission ({word_count_col})", 0, 1000, 50, disabled=True)

    # MOVED: Filter method and submissions sliders
    filter_method = st.radio(
        "Filter Students Based On:",
        ("Total Submissions", "Baseline Submissions"),
        index=0,
        key='filter_method_toggle',
        horizontal=True,
        help="Choose whether to filter students by their total submissions or by the number of baseline submissions.",
        disabled=df is None
    )

    min_submissions = st.slider(
        "Min Total Submissions per Student",
        min_value=1, max_value=20, value=3, step=1,
        help="Filter students: Only include students with at least this many *total* submissions after initial filters.",
        disabled=(df is None or filter_method != "Total Submissions")
    )

    min_baseline_submissions = st.slider(
        f"Min Baseline Submissions per Student (for '{selected_metric or 'selected metric'}')",
        min_value=1, max_value=10, value=2, step=1,
        help=f"Filter analysis: Only include deviations from students whose baseline for '{selected_metric or 'N/A'}' was calculated using at least this many submissions.",
        disabled=(df is None or filter_method != "Baseline Submissions" or selected_metric is None or selected_metric == "no_metric_available")
    )

    # Student focus dropdown
    unique_students_options = ["All Students (Aggregate View)"] + all_student_numbers
    focus_student = st.selectbox(
        "Focus on Specific Student (Optional)",
        options=unique_students_options,
        index=0,
        disabled=df is None,
        key="student_focus_dropdown_main"
    )

    # --- Histogram Bin Control ---
    st.subheader("6. Visualization Options")
    num_bins = st.slider(
        "Number of Bins for Deviation Histogram",
        min_value=5,
        max_value=100,
        value=30,
        step=1,
        help="Adjust the granularity of the deviation distribution histogram.",
        disabled=df is None
    )

    # --- Outlier Filtering ---
    st.subheader("7. Outlier Filtering")
    filter_outliers = st.checkbox(
        "Filter Outliers based on Z-score Deviation",
        value=False,
        help="If checked and Deviation Method is 'z_score', removes points where |z-score| > threshold.",
        disabled=(df is None or deviation_method != 'z_score')
    )
    max_z_slider_limit = st.session_state.get('max_abs_z_score_limit', 5.0)

    z_score_threshold = st.slider(
        "Z-score Outlier Threshold (|Z| <= value)",
        min_value=1.0,
        max_value=max_z_slider_limit,
        value=min(3.0, max_z_slider_limit),
        step=0.1,
        help="Set the absolute z-score deviation threshold for outlier removal. Max adjusts based on data.",
        disabled=(not filter_outliers or deviation_method != 'z_score')
    )


# --- Main Panel --- Data Processing and Visualization ---

if df is None:
    st.info("👈 Please upload a CSV file using the sidebar.")
else:
    # --- Apply Filters Sequentially ---
    df_filtered = df.copy()
    initial_row_count = len(df_filtered)
    rows_removed_log = []

    # 1. Filter by Null AI Score
    if exclude_null_ai and ai_scores_present and ai_score_col is not None:
        rows_before = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=[ai_score_col])
        rows_removed = rows_before - len(df_filtered)
        if rows_removed > 0:
            rows_removed_log.append(f"- {rows_removed} rows with missing '{ai_score_col}'.")

    # 2. Filter by Word Count
    if not word_count_slider_disabled and word_count_col in df_filtered.columns:
        rows_before = len(df_filtered)
        # Ensure comparison is safe
        numeric_wc = pd.to_numeric(df_filtered[word_count_col], errors='coerce')
        df_filtered = df_filtered[numeric_wc >= min_word_count]
        rows_removed = rows_before - len(df_filtered)
        if rows_removed > 0:
            rows_removed_log.append(f"- {rows_removed} rows below min word count ({min_word_count}).")

    # 2b. Filter by selected metric range
    if metric_range is not None and selected_metric != "no_metric_available" and selected_metric in df_filtered.columns:
        # Check if metric column is numeric before filtering
        if pd.api.types.is_numeric_dtype(df_filtered[selected_metric]):
            rows_before = len(df_filtered)
            low_val, high_val = metric_range
            # Ensure comparison is safe with potential NaNs
            numeric_metric = pd.to_numeric(df_filtered[selected_metric], errors='coerce')
            df_filtered = df_filtered[numeric_metric.between(low_val, high_val, inclusive='both')]
            rows_removed = rows_before - len(df_filtered)
            if rows_removed > 0:
                rows_removed_log.append(f"- {rows_removed} rows outside '{selected_metric}' range ({low_val:.2f} - {high_val:.2f}).")
        else:
            st.warning(f"Metric '{selected_metric}' is not numeric. Cannot apply range filter.")
            # Optionally disable metric range filter if non-numeric? Or just skip.

    # --- Update Dynamic Filter Options --- dynamically in sidebar (complex)
    # Simpler approach: Filter data first, then create sidebar widgets based on filtered data
    # (Requires careful state management or re-creation)
    # Current code applies filters based on initial sidebar state. Re-run needed for updates.
    
    # 3. Apply Subject Filters (using the selection from the initial multiselect)
    if selected_subjects_user and 'Subject' in df_filtered.columns:
        rows_before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['Subject'].isin(selected_subjects_user)]
        rows_removed = rows_before - len(df_filtered)
        if rows_removed > 0:
            rows_removed_log.append(f"- {rows_removed} rows excluded by Subject filter.")

    # 4. Apply Subject Category Filters (using the selection from the initial multiselect)
    if selected_categories_user and 'Subject Category' in df_filtered.columns:
        rows_before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['Subject Category'].isin(selected_categories_user)]
        rows_removed = rows_before - len(df_filtered)
        if rows_removed > 0:
            rows_removed_log.append(f"- {rows_removed} rows excluded by Category filter.")

    # 5. Filter by Min submissions per student
    students_removed_by_count_filter = 0
    if filter_method == "Total Submissions":
        if 'student_number' in df_filtered.columns:
            rows_before_filter = len(df_filtered)
            student_counts = df_filtered['student_number'].value_counts()
            students_to_keep = student_counts[student_counts >= min_submissions].index
            initial_students_count = df_filtered['student_number'].nunique()
            df_filtered = df_filtered[df_filtered['student_number'].isin(students_to_keep)]
            rows_removed_filter = rows_before_filter - len(df_filtered)
            final_students_count = df_filtered['student_number'].nunique()
            students_removed_by_count_filter = initial_students_count - final_students_count
            if rows_removed_filter > 0:
                rows_removed_log.append(f"- {rows_removed_filter} rows from {students_removed_by_count_filter} students with < {min_submissions} total submissions.")
    else:
            st.warning("'student_number' column missing. Cannot apply min total submissions filter.")
    # Baseline submission filter applied later

    # --- Display Filtering Summary ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtering Results")
    if rows_removed_log:
        st.sidebar.write("Rows removed:")
        for log_entry in rows_removed_log:
            st.sidebar.write(log_entry)
    st.sidebar.write(f"**{len(df_filtered)}** rows remaining out of {initial_row_count}.")
    st.sidebar.write(f"**{df_filtered['student_number'].nunique()}** unique students remaining.")

    # --- Distribution Plot --- 
    if selected_metric != "no_metric_available" and selected_metric in df_filtered.columns:
        st.header(f"Overall Distribution for: `{selected_metric}`")
        metric_values = df_filtered[selected_metric].dropna()
        
        if not metric_values.empty and pd.api.types.is_numeric_dtype(metric_values):
            metric_mean = metric_values.mean()
            metric_median = metric_values.median()
            metric_std = metric_values.std()
            
            fig_dist = px.histogram(
                df_filtered, 
                x=selected_metric,
                marginal="box",
                histnorm="probability density",
                title=f"Distribution of '{selected_metric}' across {len(metric_values)} submissions",
                hover_data=['student_number', 'exam_title', 'Subject', ai_score_col] if all(col in df_filtered.columns for col in ['student_number', 'exam_title', 'Subject', ai_score_col]) else None
            )
            
            fig_dist.add_vline(x=metric_mean, line_dash="solid", line_color="green", 
                       annotation_text=f"Mean: {metric_mean:.2f}", annotation_position="top right")
            fig_dist.add_vline(x=metric_median, line_dash="dash", line_color="blue", 
                       annotation_text=f"Median: {metric_median:.2f}", annotation_position="top left")
            
            if not pd.isna(metric_std) and metric_std > 0:
                 fig_dist.add_vline(x=metric_mean + metric_std, line_dash="dot", line_color="red", 
                            annotation_text="+1σ", annotation_position="bottom right")
                 fig_dist.add_vline(x=metric_mean - metric_std, line_dash="dot", line_color="red", 
                            annotation_text="-1σ", annotation_position="bottom left")
            
            fig_dist.update_layout(
                xaxis_title=selected_metric,
                yaxis_title="Probability Density",
                bargap=0.1,
                height=500,
                margin=dict(t=100, b=50),
                boxmode="group",
                boxgap=0.3
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Count", f"{len(metric_values)}")
            col2.metric("Mean", f"{metric_mean:.2f}")
            col3.metric("Median", f"{metric_median:.2f}")
            col4.metric("Std Dev", f"{metric_std:.2f}" if not pd.isna(metric_std) else "N/A")
            col5.metric("Range", f"{metric_values.min():.2f} to {metric_values.max():.2f}")
        elif not metric_values.empty: # Non-numeric
            st.warning(f"Cannot plot distribution for non-numeric metric '{selected_metric}'.")
        else: # Empty after filtering/dropna
            st.warning(f"No valid data points available for '{selected_metric}' after applying filters.")
    
    # --- Final Check and Analysis --- 
    if df_filtered.empty:
        st.warning("No data remaining after applying all filters. Adjust filters in the sidebar.")
        st.stop()
    # Check AI threshold validity
    elif ai_score_col is None:
        st.error("No valid AI Score column selected or found. Cannot proceed.")
        st.stop()
    elif ai_scores_present and low_ai_threshold >= high_ai_threshold:
        st.error("Low AI threshold must be strictly less than High AI threshold. Adjust sliders.")
        st.stop()
    else:
        # Proceed with analysis only if we have data and valid thresholds
        df_processed = add_ai_category(df_filtered, ai_score_col, low_ai_threshold, high_ai_threshold)

        students_with_low = df_processed[df_processed['ai_category'] == 'Low (Baseline)']['student_number'].unique()
        students_with_high = df_processed[df_processed['ai_category'] == 'High (Target)']['student_number'].unique()
        target_students = list(set(students_with_low) & set(students_with_high))

        if not target_students:
            st.warning("No students found with both 'Low (Baseline)' and 'High (Target)' submissions after filters. Consider adjusting filters.")
            if st.checkbox("Show Filtered Data Table (No Baseline/Target Pairs Found)"):
                st.subheader("Filtered Data")
                display_cols_basic = [col for col in ['student_number', 'exam_id', 'exam_title', 'Subject', 'Subject Category', ai_score_col, 'ai_category', selected_metric] if col in df_processed.columns]
                other_ai_col = 'average_ai_score_across_all_questions' if ai_score_col == 'max_ai_score_of_all_questions' else 'max_ai_score_of_all_questions'
                if other_ai_col in df_processed.columns and other_ai_col not in display_cols_basic:
                    ai_score_index = display_cols_basic.index(ai_score_col) if ai_score_col in display_cols_basic else -1
                    if ai_score_index != -1:
                        display_cols_basic.insert(ai_score_index + 1, other_ai_col)
                st.dataframe(df_processed[display_cols_basic])
            st.stop()
        else:
            st.info(f"Analyzing {len(target_students)} students with both baseline and target submissions.")

            metrics_to_analyze = [selected_metric] if selected_metric != "no_metric_available" else []
            if not metrics_to_analyze:
                st.error("No valid metric selected for analysis! Please select one.")
                st.stop()
            elif selected_metric not in df_processed.columns or not pd.api.types.is_numeric_dtype(df_processed[selected_metric]):
                st.error(f"Selected metric '{selected_metric}' is not numeric or not found in processed data.")
                st.stop()
            else:
                # Calculate Baselines
                baseline_stats = calculate_baselines(df_processed[df_processed['student_number'].isin(target_students)],
                                                    ai_score_col, low_ai_threshold, metrics_to_analyze)

                if not baseline_stats:
                    st.warning("Could not calculate baselines. Ensure enough 'Low AI' submissions exist.")
                    st.stop()
                else:
                    # Calculate Deviations
                    high_ai_subs = df_processed[
                        (df_processed['student_number'].isin(target_students)) & 
                        (df_processed['ai_category'] == 'High (Target)')
                    ].copy()

                    if high_ai_subs.empty:
                        st.warning("No 'High (Target)' submissions found for students with baseline data.")
                        st.stop()
                    else:
                        deviation_col_name = f'{selected_metric}_deviation'
                        high_ai_subs[deviation_col_name] = high_ai_subs.apply(
                            lambda row: calculate_deviation(row, baseline_stats, selected_metric, deviation_method), axis=1
                        )
                        valid_deviations = high_ai_subs.dropna(subset=[deviation_col_name])

                        if valid_deviations.empty:
                            st.warning(f"Could not calculate valid deviations for '{selected_metric}'. Check data and baseline calculation.")
                            st.stop()
                        else:
                            # Apply Min Baseline Submissions Filter
                            rows_before_baseline_filter = len(valid_deviations)
                            students_before_baseline_filter = valid_deviations['student_number'].nunique()
                            students_removed_by_baseline_count = 0

                            if filter_method == "Baseline Submissions":
                                def check_baseline_count(student):
                                    count = get_baseline_stat(student, 'count', baseline_stats, selected_metric)
                                    return not pd.isna(count) and count >= min_baseline_submissions
                                
                                initial_students = valid_deviations['student_number'].unique()
                                valid_deviations = valid_deviations[valid_deviations['student_number'].apply(check_baseline_count)]
                                final_students = valid_deviations['student_number'].unique()
                                students_removed_by_baseline_count = len(initial_students) - len(final_students)
                                rows_removed_baseline = rows_before_baseline_filter - len(valid_deviations)
                                
                                if rows_removed_baseline > 0:
                                    st.caption(f"ℹ️ {rows_removed_baseline} high-AI rows removed ({students_removed_by_baseline_count} students) due to < {min_baseline_submissions} baseline submissions for '{selected_metric}'.")

                            # Check if empty AFTER baseline filter
                            if valid_deviations.empty:
                                st.warning(f"No deviation data remaining after applying the minimum baseline submissions filter (Min: {min_baseline_submissions} for '{selected_metric}'). Adjust filters.")
                                st.stop()

                            # Update Max Z for Slider
                            if deviation_method == 'z_score' and not valid_deviations.empty:
                                current_max_abs_z = valid_deviations[deviation_col_name].abs().max()
                                if not pd.isna(current_max_abs_z):
                                    new_max_z_slider_limit = max(1.1, np.ceil(current_max_abs_z * 1.1))
                                    st.session_state['max_abs_z_score_limit'] = new_max_z_slider_limit
                            elif deviation_method != 'z_score':
                                st.session_state['max_abs_z_score_limit'] = 5.0 # Reset

                            # Outlier Filtering
                            rows_before_outlier_filter = len(valid_deviations)
                            initial_valid_deviations = valid_deviations.copy()
                            outliers_removed = 0
                            if filter_outliers and deviation_method == 'z_score':
                                valid_deviations = valid_deviations[valid_deviations[deviation_col_name].abs() <= z_score_threshold]
                                outliers_removed = rows_before_outlier_filter - len(valid_deviations)
                                if outliers_removed > 0:
                                    st.caption(f"ℹ️ {outliers_removed} outlier(s) removed (|Z-score| > {z_score_threshold:.1f}).")

                            # Re-check if empty after outlier filtering
                            if valid_deviations.empty:
                                st.warning(f"No data remaining after Z-score outlier filter. Adjust threshold or disable filter.")
                                if outliers_removed > 0 and st.button("Show Plot Before Outlier Filter"):
                                    valid_deviations = initial_valid_deviations # Restore for plot
                                else:
                                    st.stop()
                            
                            # --- Start Plotting and Display --- 
                            st.header(f"Analysis for Metric: `{selected_metric}`")

                            # Deviation Distribution Plot
                            st.subheader(f"Deviation of High AI Submissions (Method: {deviation_method})")
                            st.markdown(f"""
                            Distribution of deviations for `{selected_metric}`. Each point represents a 'High (Target)' submission,
                            showing its difference from that student's baseline ('Low') submissions.
                            """)

                            fig_dev = px.histogram(valid_deviations, x=deviation_col_name,
                                                title=f"Distribution of '{selected_metric}' Deviations",
                                                labels={deviation_col_name: f"Deviation ({deviation_method})"},
                                                marginal="box",
                                                hover_data=[col for col in ['student_number', 'exam_title', 'Subject', 'Subject Category', ai_score_col, selected_metric] if col in valid_deviations.columns],
                                                nbins=num_bins
                                                )
                            fig_dev.update_layout(bargap=0.1)
                            st.plotly_chart(fig_dev, use_container_width=True)

                            st.markdown(f"""
                            *   **Z-score:** Values far from 0 (e.g., > |2|) indicate significant deviation.
                            *   **Raw Difference:** Shows absolute difference from the student's average baseline.
                            *   **Percent Difference:** Shows percentage change relative to the student's average baseline.
                            """)

                            # Single Student View
                            final_student_list_for_analysis = sorted(df_processed['student_number'].unique().tolist())
                            is_focus_student_valid = (focus_student != "All Students (Aggregate View)") and (focus_student in final_student_list_for_analysis)

                            if focus_student != "All Students (Aggregate View)":
                                if is_focus_student_valid:
                                    st.subheader(f"Detailed View for Student: {focus_student}")
                                    student_df = df_processed[df_processed['student_number'] == focus_student].copy()

                                    # Ensure deviation calculation happens for this student's data
                                    student_df_deviation_col = f'{selected_metric}_deviation'
                                    # Check if baseline exists before calculating deviation
                                    if focus_student in baseline_stats:
                                        student_df[student_df_deviation_col] = student_df.apply(
                                            lambda row: calculate_deviation(row, baseline_stats, selected_metric, deviation_method), axis=1
                                        )
                                    else:
                                        student_df[student_df_deviation_col] = np.nan # No baseline, no deviation

                                    fig_student = go.Figure()

                                    # Add points for different AI categories
                                    for category, color, symbol, size in [
                                        ('Low (Baseline)', 'blue', 'circle', 8),
                                        ('High (Target)', 'red', 'x', 10),
                                        ('Mid (Ignored)', 'grey', 'diamond', 6)
                                    ]:
                                        cat_data = student_df[student_df['ai_category'] == category]
                                        if not cat_data.empty:
                                            hover_texts = []
                                            for _, row in cat_data.iterrows():
                                                text = f"Exam: {row.get('exam_title', 'N/A')}<br>Subject: {row.get('Subject', 'N/A')}<br>{ai_score_col}: {row.get(ai_score_col, np.nan):.2f}"
                                                if category == 'High (Target)' and student_df_deviation_col in row:
                                                    dev_val = row[student_df_deviation_col]
                                                    text += f"<br>Dev: {dev_val:.2f}" if not pd.isna(dev_val) else "<br>Dev: N/A"
                                                hover_texts.append(text)
                                            
                                            fig_student.add_trace(go.Scatter(
                                                x=cat_data['exam_id'], y=cat_data[selected_metric],
                                                mode='markers', name=category,
                                                marker=dict(color=color, size=size, symbol=symbol),
                                                text=hover_texts,
                                                hoverinfo='text+y'
                                            ))

                                    # Add baseline lines if available
                                    if focus_student in baseline_stats and f'{selected_metric}_mean' in baseline_stats[focus_student]:
                                        mean_val = baseline_stats[focus_student][f'{selected_metric}_mean']
                                        std_val = baseline_stats[focus_student].get(f'{selected_metric}_std', np.nan)
                                        count_val = baseline_stats[focus_student].get(f'{selected_metric}_count', 0)

                                        if not pd.isna(mean_val):
                                            fig_student.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text=f"Baseline Mean ({int(count_val)} exams)")
                                            if not pd.isna(std_val) and std_val > 1e-9:
                                                fig_student.add_hline(y=mean_val + 2*std_val, line_dash="dot", line_color="orange", annotation_text="+2 Std Dev")
                                                fig_student.add_hline(y=mean_val - 2*std_val, line_dash="dot", line_color="orange", annotation_text="-2 Std Dev")

                                    fig_student.update_layout(
                                        title=f"{selected_metric} Trend for Student {focus_student}",
                                        xaxis_title="Exam ID (Not necessarily chronological)",
                                        yaxis_title=selected_metric,
                                        hovermode="closest"
                                    )
                                    st.plotly_chart(fig_student, use_container_width=True)

                                    st.write("Data for selected student:")
                                    display_cols = [col for col in ['exam_id', 'exam_title', 'Subject', 'Subject Category', ai_score_col, 'ai_category', selected_metric, student_df_deviation_col] if col in student_df.columns]
                                    other_ai_col = 'average_ai_score_across_all_questions' if ai_score_col == 'max_ai_score_of_all_questions' else 'max_ai_score_of_all_questions'
                                    if other_ai_col in student_df.columns and other_ai_col not in display_cols:
                                        ai_score_index = display_cols.index(ai_score_col) if ai_score_col in display_cols else -1
                                        if ai_score_index != -1:
                                            display_cols.insert(ai_score_index + 1, other_ai_col)
                                    st.dataframe(student_df[display_cols].sort_values(by='exam_id'))
                                else:
                                    st.warning(f"Student '{focus_student}' selected, but data removed by filters. Choose another student or adjust filters.")

                            # Summary Statistics Table
                            st.subheader(f"Summary Statistics for Deviations ({deviation_method}) - {len(valid_deviations)} High AI points from {valid_deviations['student_number'].nunique()} students")
                            st.dataframe(valid_deviations[deviation_col_name].describe())

                            # Optional Raw Data Table
                            if st.checkbox("Show Full Processed Data Table Used for Analysis"):
                                st.subheader(f"Data for {len(target_students)} students with baseline/target pairs")
                                display_cols_processed = [col for col in ['student_number', 'exam_id', 'exam_title', 'Subject', 'Subject Category', ai_score_col, 'ai_category', selected_metric] if col in df_processed.columns]
                                other_ai_col = 'average_ai_score_across_all_questions' if ai_score_col == 'max_ai_score_of_all_questions' else 'max_ai_score_of_all_questions'
                                if other_ai_col in df_processed.columns and other_ai_col not in display_cols_processed:
                                    ai_score_index = display_cols_processed.index(ai_score_col) if ai_score_col in display_cols_processed else -1
                                    if ai_score_index != -1:
                                        display_cols_processed.insert(ai_score_index + 1, other_ai_col)
                                st.dataframe(df_processed.sort_values(by='student_number')[display_cols_processed])
                                
                                st.markdown("---")
                                st.write(f"Calculated Deviations (Filtered - {len(valid_deviations)} High AI submissions from {valid_deviations['student_number'].nunique()} students):")

                                # Create series for baseline stats display
                                baseline_stats_display = pd.DataFrame({
                                    'student_number': valid_deviations['student_number'].unique()
                                })
                                for stat in ['mean', 'count', 'min', 'p20', 'p80', 'max']:
                                    stat_key = f'{selected_metric}_{stat}'
                                    baseline_stats_display[f'baseline_{stat_key}'] = baseline_stats_display['student_number'].apply(
                                        lambda s: baseline_stats.get(s, {}).get(stat_key, np.nan)
                                    )
                                # Merge baseline stats back for display
                                df_display_deviations = pd.merge(valid_deviations, baseline_stats_display, on='student_number', how='left')

                                # Define columns to display
                                baseline_cols_to_show = [f'baseline_{selected_metric}_{stat}' for stat in ['min', 'p20', 'mean', 'p80', 'max', 'count']]
                                display_cols_deviation = [
                                    'student_number', 'exam_id', 'Subject',
                                    ai_score_col, selected_metric
                                ] + baseline_cols_to_show + [deviation_col_name]
                                
                                other_ai_col = 'average_ai_score_across_all_questions' if ai_score_col == 'max_ai_score_of_all_questions' else 'max_ai_score_of_all_questions'
                                if other_ai_col in df_display_deviations.columns:
                                    ai_score_index = display_cols_deviation.index(ai_score_col) if ai_score_col in display_cols_deviation else -1
                                    if ai_score_index != -1:
                                        display_cols_deviation.insert(ai_score_index + 1, other_ai_col)

                                final_display_cols = [col for col in display_cols_deviation if col in df_display_deviations.columns]
                                st.dataframe(df_display_deviations[final_display_cols])

                            # Baseline Configuration Summary
                            st.markdown("---")
                            st.subheader("📋 Current Baseline Configuration")
                            metric_range_text = f"{metric_range[0]:.2f} - {metric_range[1]:.2f}" if metric_range is not None else "Full range"
                            min_submissions_text = f"Min {min_baseline_submissions} baseline exams" if filter_method == "Baseline Submissions" else f"Min {min_submissions} total submissions"
                            
                            config_md = f"""
                            | Parameter        | Value                                            |
                            |------------------|--------------------------------------------------|
                            | **Metric**       | `{selected_metric}`                              |
                            | **Max AI Score** | `{ai_score_col}` ≤ {low_ai_threshold:.2f}          |
                            | **Min Exams**    | {min_submissions_text}                           |
                            | **Metric Range** | {metric_range_text}                              |
                            | **Min Word Count** | {min_word_count}                                 |
                            """
                            st.markdown(config_md)
                            
                            if metric_range is not None and abs(metric_range[0] - lower_fence) < 0.01 and abs(metric_range[1] - upper_fence) < 0.01:
                                st.caption("Note: Metric range matches default box plot whiskers (Q1-1.5×IQR to Q3+1.5×IQR).")
                                    
                            # Proposed AI Flag Threshold Section
                            st.markdown("### Proposed AI Flag Threshold")
                            st.markdown("<hr style='margin-top: 0; margin-bottom: 1em; border-width: 3px; border-color: #f63366;'>", unsafe_allow_html=True)
                            
                            z_score_flag_threshold_input = st.text_input(
                                "Absolute Z-Score",
                                value="2.0",
                                help="Specify the absolute Z-score threshold for flagging submissions."
                            )
                            
                            if st.button("Save Configuration"):
                                try:
                                    # Validate Z-score input
                                    z_score_flag_threshold = float(z_score_flag_threshold_input)
                                    
                                    save_dir = "user_analysis"
                                    if not os.path.exists(save_dir):
                                        os.makedirs(save_dir)
                                    
                                    config_data = {
                                        "metric": selected_metric,
                                        "baseline": {
                                            "ai_score_type": ai_score_col,
                                            "low_ai_threshold": low_ai_threshold,
                                            "min_exams_filter_type": filter_method,
                                            "min_exams_threshold": min_baseline_submissions if filter_method == "Baseline Submissions" else min_submissions,
                                            "metric_range_min": metric_range[0] if metric_range is not None else None,
                                            "metric_range_max": metric_range[1] if metric_range is not None else None,
                                            "min_word_count": min_word_count
                                        },
                                        "proposed_cheating_threshold": {
                                            "calculation_method": "absolute z-score",
                                            "threshold": z_score_flag_threshold,
                                            "cheating_flagged_if": "greater_than"
                                        },
                                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    filename = f"Analysis_Threshold_{selected_metric}.json"
                                    filepath = os.path.join(save_dir, filename)
                                    
                                    with open(filepath, "w") as f:
                                        json.dump(config_data, f, indent=4)
                                    
                                    st.success(f"Configuration saved to {filepath}")
                                except ValueError:
                                    st.error("Invalid input for Absolute Z-Score. Please enter a numeric value.")
                                except Exception as e:
                                    st.error(f"Failed to save configuration: {e}")