import streamlit as st
import pickle
import numpy as np
import os

scaler = pickle.load(open("scaler.sav", 'rb'))
model = pickle.load(open("model.sav", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0
    
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score, total_score, average_score]])
    
    scaled_features = scaler.transform(feature_array)
    probabilities = model.predict_proba(scaled_features)
    
    #five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs

st.set_page_config(page_title="Career Recommendations", page_icon="💼", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f4f7f9;
    }
    h1 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .highlight-box {
        background-color: #e1f5fe;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        color: #0d47a1;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Career Recommendations By Krishna 💼")
st.subheader("Get customized career suggestions based on your profile!")

gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.checkbox("Do you have a part-time job?")
absence_days = st.number_input("Number of absent days", min_value=0, max_value=365, step=1)
extracurricular_activities = st.checkbox("Are you involved in extracurricular activities?")
weekly_self_study_hours = st.number_input("Weekly self-study hours", min_value=0.0, max_value=168.0, step=0.5)

st.subheader("Academic Scores out of 100")
math_score = st.number_input("Math Score", min_value=0, max_value=100, step=1)
history_score = st.number_input("History Score", min_value=0, max_value=100, step=1)
physics_score = st.number_input("Physics Score", min_value=0, max_value=100, step=1)
chemistry_score = st.number_input("Chemistry Score", min_value=0, max_value=100, step=1)
biology_score = st.number_input("Biology Score", min_value=0, max_value=100, step=1)
english_score = st.number_input("English Score", min_value=0, max_value=100, step=1)
geography_score = st.number_input("Geography Score", min_value=0, max_value=100, step=1)

total_score = math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score
average_score = total_score / 7
st.write(f"**Total Score**: {total_score}")
st.write(f"**Average Score**: {average_score:.2f}")

if st.button("Get Career Recommendations"):
    with st.spinner('Analyzing your profile...'):
        recommendations = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                          weekly_self_study_hours, math_score, history_score, physics_score,
                                          chemistry_score, biology_score, english_score, geography_score,
                                          total_score, average_score)
    
        st.subheader("🌟 Top Career Recommendations")
        st.subheader("Rank 1 would be to Find out what you like doing best, and get someone to pay you for doing it.")
    st.markdown('<div class="ranked-list">', unsafe_allow_html=True)
    
    for rank, (career, prob) in enumerate(recommendations, start=2):
        st.markdown(
            f'<div class="highlight-box">'
            f"Rank {rank}: {career} - Probability: {prob * 100:.2f}%"
            f'</div>',
            unsafe_allow_html=True
        )
