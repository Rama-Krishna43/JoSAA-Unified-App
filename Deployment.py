import streamlit as st
import pandas as pd
import joblib
import pickle
import altair as alt
import warnings
warnings.filterwarnings('ignore')

from streamlit.components.v1 import html

verification_code="yRH8JcxZlFPIRId2blA6DD9Z4z42LdTgznTQLI3PIWI"

st.components.v1.html(
    """
    <head>
    <meta name="google-site-verification" content="YOUR_CODE" />
    </head>
    """,
    height=0,
)


# --- 1. SET PAGE CONFIG (Must be the first st command) ---
st.set_page_config(
    page_title="JoSAA 2026 Premium Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Rama-Krishna43/JoSAA-College-Predictor',
        'Report a bug': "https://github.com/Rama-Krishna43/JoSAA-College-Predictor/issues",
        'About': "# JoSAA 2026 Advanced College Predictor & Eligibility Suite"
    }
)

# --- 2. LOAD DATA & MODELS ---
@st.cache_resource
def load_predictor_assets():
    model = joblib.load('college_predictor_compressed.joblib')
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    df_2023 = pd.read_csv('josaa_2023_all_institutes_full.csv')
    df_2024 = pd.read_csv('josaa_2024_reparsed_all_institutes.csv')
    df_2025 = pd.read_csv('josaa_2025_opening_closing_ranks_all_institutes.csv')
    
    df_2023['Year'] = 2023
    df_2024['Year'] = 2024
    df_2025['Year'] = 2025
    
    df = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
    df.dropna(subset=['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender'], inplace=True)
    
    df['Opening Rank'] = pd.to_numeric(df['Opening Rank'].astype(str).str.replace('P', ''), errors='coerce')
    df['Closing Rank'] = pd.to_numeric(df['Closing Rank'].astype(str).str.replace('P', ''), errors='coerce')
    df.dropna(subset=['Opening Rank', 'Closing Rank'], inplace=True)
    
    return model, encoders, df

@st.cache_data
def load_eligibility_data(pickle_file="josaa_data.pkl"):
    try:
        return pd.read_pickle(pickle_file)
    except FileNotFoundError:
        return pd.DataFrame()

try:
    model, encoders, df_predict = load_predictor_assets()
except Exception as e:
    st.error(f"Error loading predictor models/data: {e}")
    model, encoders, df_predict = None, None, pd.DataFrame()

df_eligibility = load_eligibility_data()

# --- 3. PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    /* Global Fonts & Backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title */
    .premium-title {
        background: -webkit-linear-gradient(45deg, #4F46E5, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .premium-subtitle {
        text-align: center;
        color: #6B7280;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.2em;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
    }
    
    /* Cards and Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }
    
    /* Radio buttons */
    div[role="radiogroup"] > label {
        padding: 10px 20px;
        background: #f3f4f6;
        border-radius: 8px;
        margin-right: 10px;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    div[role="radiogroup"] > label:hover {
        background: #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. CORE LOGIC FUNCTIONS ---
def get_eligible_programs(df, user_exam, my_rank, my_seat_type, my_gender):
    if user_exam.lower() == 'advanced':
        institute_filter = df['Institute Name'].str.contains('Indian Institute of Technology', na=False)
        quota_filter = (df['Quota'] == 'AI')
    elif user_exam.lower() == 'mains':
        institute_filter = ~df['Institute Name'].str.contains('Indian Institute of Technology', na=False)
        quota_filter = (df['Quota'].isin(['AI', 'HS', 'OS', 'GO', 'JK', 'LA']))
    else:
        return pd.DataFrame() 

    filtered_df = df[institute_filter & quota_filter].copy()

    eligible_programs = filtered_df[
        (filtered_df['Seat Type'] == my_seat_type) &
        (filtered_df['Gender'] == my_gender) &
        (pd.to_numeric(filtered_df['Closing Rank'], errors='coerce') >= my_rank)
    ]

    columns_to_show = [
        'Institute Name', 
        'Academic Program Name', 
        'Quota', 
        'Seat Type', 
        'Opening Rank',
        'Closing Rank'
    ]
    
    eligible_programs_final = eligible_programs.sort_values(
        by=['Institute Name', 'Closing Rank']
    )[columns_to_show]
    
    return eligible_programs_final

# --- 5. PAGE RENDERERS ---

def show_predictor():
    st.markdown("<h2 style='text-align:center;'>🔮 College Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Predict your admission chances for a specific institute and program using our ML model.</p>", unsafe_allow_html=True)
    
    if df_predict.empty:
        st.warning("Predictor data could not be loaded.")
        return

    st.markdown("---")
    inst_type = st.radio("Select Institute Category", ["IIT (Indian Institute of Technology)", "Non-IIT (NITs, IIITs, GFTIs)"], horizontal=True)

    if "Non-IIT" in inst_type:
        mask = df_predict['Institute'].str.contains('Indian Institute of Technology') & ~df_predict['Institute'].str.contains('Information')
        display_df = df_predict[~mask]
    else:
        mask = df_predict['Institute'].str.contains('Indian Institute of Technology') & ~df_predict['Institute'].str.contains('Information')
        display_df = df_predict[mask]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Personal Details")
        user_rank = st.number_input('Enter your Category Rank', min_value=1, max_value=400000, value=10000, step=100)
        
        available_institutes = sorted(display_df['Institute'].unique())
        selected_institute = st.selectbox('Select Institute', available_institutes)
        
        if not selected_institute: return
        df_filtered_1 = display_df[display_df['Institute'] == selected_institute]
        available_programs = sorted(df_filtered_1['Academic Program Name'].unique())
        selected_program = st.selectbox('Select Academic Program', available_programs)

    with col2:
        st.subheader("🎯 Preferences")
        if not selected_program: return
        df_filtered_2 = df_filtered_1[df_filtered_1['Academic Program Name'] == selected_program]
        selected_quota = st.selectbox('Select Quota', sorted(df_filtered_2['Quota'].unique()))
        
        df_filtered_3 = df_filtered_2[df_filtered_2['Quota'] == selected_quota]
        selected_seat_type = st.selectbox('Select Seat Type', sorted(df_filtered_3['Seat Type'].unique()))
        
        df_filtered_4 = df_filtered_3[df_filtered_3['Seat Type'] == selected_seat_type]
        selected_gender = st.selectbox('Select Gender', sorted(df_filtered_4['Gender'].unique()))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('Predict My Chance 🚀'):
        with st.spinner('Analyzing historical data...'):
            try:
                input_data = {
                    'Institute': encoders['Institute'].transform([selected_institute])[0],
                    'Academic Program Name': encoders['Academic Program Name'].transform([selected_program])[0],
                    'Quota': encoders['Quota'].transform([selected_quota])[0],
                    'Seat Type': encoders['Seat Type'].transform([selected_seat_type])[0],
                    'Gender': encoders['Gender'].transform([selected_gender])[0],
                    'Year': 2026
                }
                input_df = pd.DataFrame([input_data])
                predicted_rank = model.predict(input_df)[0]
                predicted_rank = int(round(predicted_rank))

                st.markdown("### 📊 Prediction Result")
                if user_rank <= predicted_rank:
                    st.success(f"🎉 **High Chance!** Predicted Closing Rank is **{predicted_rank:,}**. Your rank ({user_rank:,}) is within the safe range.")
                else:
                    st.error(f"😞 **Low Chance.** Predicted Closing Rank is **{predicted_rank:,}**. Your rank ({user_rank:,}) might not make the cut.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

def show_eligibility():
    st.markdown("<h2 style='text-align:center;'>🎯 Eligibility Finder</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Discover all programs you are eligible for across India based on your rank.</p>", unsafe_allow_html=True)
    st.info("ℹ️ These eligibility results are based on 2025 data only.")
    
    if df_eligibility.empty:
        st.warning("Eligibility dataset `josaa_data.pkl` not found. Please ensure it is present in the directory.")
        return

    st.markdown("---")
    
    seat_types = sorted(df_eligibility['Seat Type'].dropna().unique())
    genders = sorted(df_eligibility['Gender'].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        user_exam = st.radio("1. Exam Qualified", ('Advanced', 'Mains'), horizontal=True)
        my_seat_type = st.selectbox("3. Your Seat Type/Reservation Category", options=seat_types)
    with col2:
        my_rank = st.number_input("2. Your Rank", min_value=1, value=5000)
        my_gender = st.selectbox("4. Your Gender Eligibility", options=genders)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Find Eligible Programs 🔍"):
        with st.spinner("Searching for eligible programs..."):
            st.session_state['eligibility_results'] = get_eligible_programs(df_eligibility, user_exam, my_rank, my_seat_type, my_gender)
            st.session_state['last_searched_rank'] = my_rank

    if 'eligibility_results' in st.session_state:
        results_df = st.session_state['eligibility_results']

        if results_df.empty:
            st.warning(f"No programs found for Rank {st.session_state.get('last_searched_rank', my_rank)}.")
        else:
            st.success(f"Found {len(results_df)} eligible programs! Scroll down to view them.")
            
            # Actions: Search & Download
            act_col1, act_col2, act_col3 = st.columns([2, 1, 1])
            with act_col1:
                search_term = st.text_input("🔍 Search Institute or Program Name...", key="search_eligibility")
            
            # Filter Dataframe if search term is provided
            if search_term:
                mask = results_df['Institute Name'].str.contains(search_term, case=False, na=False) | \
                       results_df['Academic Program Name'].str.contains(search_term, case=False, na=False)
                display_df = results_df[mask]
            else:
                display_df = results_df

            @st.dialog("Eligibility Results (Expanded View)", width="large")
            def show_fullscreen_results(df):
                st.dataframe(df, height=600, use_container_width=True)

            with act_col2:
                # Add some vertical space so the button aligns with the text input
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download",
                    data=csv,
                    file_name="josaa_eligible_programs.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with act_col3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("⛶ Fullscreen", use_container_width=True):
                    show_fullscreen_results(display_df)

            st.dataframe(display_df, height=500, use_container_width=True)



# --- 6. MAIN APP LAYOUT ---
st.markdown("<h1 class='premium-title'>🎓 JoSAA 2026 Suite</h1>", unsafe_allow_html=True)
st.markdown("<p class='premium-subtitle'>Advanced Analytics, Predictions, and Eligibility Tracking</p>", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>🎓</h1>", unsafe_allow_html=True)
    st.markdown("### Navigation")
    selection = st.radio("Go to", ["🔮 College Predictor", "🎯 Eligibility Finder"])
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This application unifies JoSAA College prediction and eligibility finding in one place. Using historical data (2023-2025).")

# Route to proper view
if selection == "🔮 College Predictor":
    show_predictor()
elif selection == "🎯 Eligibility Finder":
    show_eligibility()

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#9ca3af; font-size:0.9rem;'>Built with ❤️ by Rama-Krishna43 | <a href='https://github.com/Rama-Krishna43' style='color:#6366f1; text-decoration:none;'>GitHub</a></p>", unsafe_allow_html=True)
