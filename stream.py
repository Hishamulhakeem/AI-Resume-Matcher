import streamlit as st
import joblib
import fitz  
import numpy as np

model = joblib.load('Classifier.pkl')
vectorizer = joblib.load('Vector.pkl')

allowed_categories = [
    'Mechanical Engineer', 
    'Creative ',
     'Software & IT',
     'Business & Management',
    'Electrical Engineer',
    'Civil Engineer',
    'AI/ML Engineer',
    'Computer Science'
]

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.markdown("""
    <style>
        .main-container { text-align: center; margin-top: 10px; }
        h1 { font-size: 36px; font-weight: bold; }
        .stButton button { background-color: #000; color: #fff; border-radius: 10px; }
        .stButton button:hover { background-color: #333; transform: scale(1.05); }
        .result-row { background-color: #444; padding: 10px; border-radius: 8px; margin-bottom: 5px; display: flex; justify-content: space-between; }
        .percentage-text { color: #76c7c0; font-weight: bold; }
        .rating-section { background: #333; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; }
        .disclaimer { font-size: 12px; color: #bbb; position: fixed; bottom: 10px; left: 10px; text-align: left; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container"><h1>AI Resume Matcher</h1></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your resume (Only PDF format)", type=['pdf'])

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype='pdf')
    text = ''.join(page.get_text() for page in doc)
    
    resume_keywords = ["education", "experience", "skills", "projects", "certifications", "summary", "degree", "achievements"]
    if any(keyword in text.lower() for keyword in resume_keywords) and len(text.split()) > 50:
        return text
    else:
        st.warning("⚠️ Invalid PDF: Not a resume or insufficient content.")
        return None

def rate(y_pred):
    return round(np.max(y_pred) * 100, 2)

if uploaded_file is not None:
    analyze_button = st.button("Analyze Resume")

    if analyze_button:
        resume_text = extract_text(uploaded_file)
        
        if resume_text:
            X_input = vectorizer.transform([resume_text])
            y_pred = model.predict_proba(X_input)[0]

            top_indices = np.argsort(y_pred)[-3:][::-1]
            top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices if model.classes_[i] in allowed_categories]

            st.subheader("Top Matching Jobs:")

            for category, confidence in top_predictions:
                st.markdown(f"""
                    <div class="result-row">
                        <div>{category}</div>
                        <div class="percentage-text">{confidence:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            rating = rate(y_pred) + 10
            color = '#ff4d4d' if rating < 45 else '#ffc107' if rating < 75 else '#4caf50'
            st.markdown(f"""
                <div class="rating-section">
                    <div class="rating-text" style="color: {color};">Resume Rating: {rating:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

domain_text = ", ".join(allowed_categories)
st.markdown(f"""
    <div class="disclaimer">
        <p><strong>Disclaimer Job Domains:</strong> {domain_text}</p>
    </div>
""", unsafe_allow_html=True)
