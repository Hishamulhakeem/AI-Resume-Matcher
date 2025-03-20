import streamlit as st
import joblib
import fitz  # PyMuPDF
import numpy as np

model = joblib.load('Classifier.pkl')
vectorizer = joblib.load('Vector.pkl')

allowed_domains = {
    'Computer Science ': ['Data Science', 'Java Developer', 'Python Developer', 'DotNet Developer'],
    'AI/ML Engineer' : ['AI/ML Engineer'],
    'Mechanical Engineer': ['Mechanical Engineer'],
    'Civil Engineer': ['Civil Engineer'],
    'Electrical Engineer': ['Electrical Engineering'],
    'Business & Management': ['HR', 'Business Analyst', 'Operations Manager', 'PMO'],
    'Software & IT': ['SAP Developer', 'Automation Testing', 'DevOps Engineer', 'Database', 'Hadoop', 'ETL Developer', 'Blockchain', 'Testing'],
    'Creative & Others': ['Web Designing', 'Advocate', 'Arts', 'Health and fitness', 'Sales']
}

allowed_categories = [item for sublist in allowed_domains.values() for item in sublist]

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.markdown("""
    <style>
        .main-container { text-align: center; margin-top: 30px; }
        h1 { font-size: 36px; font-weight: bold; }
        .stButton button { background: #000; color: #fff; border-radius: 10px; }
        .stButton button:hover { background: #333; transform: scale(1.05); }
        .result-row { background: #444; padding: 10px; border-radius: 8px; margin-bottom: 5px; }
        .percentage-text { color: #76c7c0; font-weight: bold; }
        .rating-section { background: #333; padding: 15px; border-radius: 8px; margin-top: 20px; }
        .disclaimer { font-size: 12px; color: #bbb; margin-top: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container"><h1>AI Resume Matcher</h1></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your resume (Only PDF format)", type=['pdf'])

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype='pdf')
    text = ''.join(page.get_text() for page in doc)
    
    if any(k in text.lower() for k in ["education", "experience", "skills", "projects"]) and len(text.split()) > 50:
        return text
    else:
        st.warning("⚠️ Invalid PDF: Not a resume or insufficient content.")
        return None

def rate(y_pred):
    return round(np.max(y_pred) * 100, 2)

if uploaded_file:
    if st.button("Analyze Resume"):
        resume_text = extract_text(uploaded_file)
        if resume_text:
            X_input = vectorizer.transform([resume_text])
            y_pred = model.predict_proba(X_input)[0]
            
            top_indices = np.argsort(y_pred)[-3:][::-1]
            top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices if model.classes_[i] in allowed_categories]

            st.subheader("Top Matching Jobs:")
            for category, confidence in top_predictions:
                st.markdown(f'<div class="result-row"><div>{category}</div><div class="percentage-text">{confidence:.2f}%</div></div>',
                            unsafe_allow_html=True)

            rating = rate(y_pred) + 10  # Boosting score by 10
            color = '#ff4d4d' if rating < 45 else '#ffc107' if rating < 75 else '#4caf50'
            st.markdown(f'<div class="rating-section"><div style="color: {color}; font-weight: bold;">Resume Rating: {rating:.2f}%</div></div>',
                        unsafe_allow_html=True)


domain_text = ", ".join(allowed_domains.keys())
st.markdown(f'<div class="disclaimer">Allowed Job Domains: {domain_text}</div>', unsafe_allow_html=True)
