import streamlit as st
import joblib
import fitz  # PyMuPDF
import numpy as np

model = joblib.load('Classifier.pkl')
vectorizer = joblib.load('Vector.pkl')
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #f4f4f4;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .main-container {
            margin-top: 30px;
            padding: 40px;
            text-align: center;
        }
        h1 {
            margin-top: 20px;
            margin-bottom: 30px;
            font-size: 36px;
            font-weight: bold;
            color: #f4f4f4;
            letter-spacing: 1px;
        }
        .stButton button {
            background-color: #000;
            color: #fff;
            padding: 10px 30px;
            border-radius: 10px;
            border: 2px solid #444;
            font-weight: bold;
            margin: 20px auto;
            display: block;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #333;
            border-color: #777;
            transform: scale(1.05);
        }
        .result-row {
            background-color: #444;
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .percentage-text {
            color: #76c7c0;
            font-weight: bold;
        }
        .rating-section {
            margin-top: 30px;
            background-color: #333;
            padding: 20px;
            border-radius: 10px;
            color: #fff;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
            font-weight: bold;
        }
        .rating-text {
            font-weight: bold;
            font-size: 18px;
        }
        .disclaimer {
            font-size: 12px;
            color: #bbb;
            position: fixed;
            bottom: 10px;
            left: 10px;
            text-align: left;
        }
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

def rate(resume_text, y_pred):
    return round(np.max(y_pred) * 100, 2)

if uploaded_file is not None:
    analyze_button = st.button("Analyze Resume")

    if analyze_button:
        resume_text = extract_text(uploaded_file)

        X_input = vectorizer.transform([resume_text])
        y_pred = model.predict_proba(X_input)[0]

        top_indices = np.argsort(y_pred)[-3:][::-1]
        top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]

        st.subheader("Top Matching Jobs:")
s
        import time
        for category, confidence in top_predictions:
            time.sleep(0.5)
            st.markdown(f"""
                <div class="result-row" style="animation: fadeIn 0.8s;">
                    <div>{category}</div>
                    <div class="percentage-text">{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

        rating = rate(resume_text, y_pred) + 10
        color = '#ff4d4d' if rating < 45 else '#ffc107' if rating < 75 else '#4caf50'
        st.markdown(f"""
        <div class="rating-section">
            <div class="rating-text" style="color: {color};">Resume Rating : {rating:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <div class="disclaimer">
        <strong>Job Domains:</strong> Mechanical Engineer, Creative, Software & IT, Business & Management, Electrical Engineer, Civil Engineer, AI/ML Engineer, Computer Science
    </div>
""", unsafe_allow_html=True)
